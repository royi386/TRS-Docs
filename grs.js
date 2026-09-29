/**
 * Structured rule books reader (General Rules, Accident Manual).
 *
 * Each book's PDF text layer is read once, split into chapters and
 * individually addressable rules (1.01, 1.02, ...), stored in SQLite with a
 * full-text index, and served to the reader page. Admin corrections live in
 * an overlay table and survive re-indexing.
 *
 * Reuses the pdf.js loading helpers from rag.js.
 */
const fs = require('fs');
const path = require('path');

let rag = null;
function getRag() {
  if (!rag) rag = require('./rag');
  return rag;
}

const BOOKS = {
  grs: {
    key: 'grs',
    label: 'General Rules',
    filename: 'GRS Full.pdf',
    tocPages: 22, // front matter + arrangement of rules, no rule bodies
    maxChapter: 18,
    titleStyle: 'grs' // "GR.1.01 Short title. –" style headings
  },
  am: {
    key: 'am',
    label: 'Accident Manual',
    filename: 'Accident_Manual_2022.pdf',
    tocPages: 17, // cover + contents listing, chapter I starts on page 18
    maxChapter: 10,
    titleStyle: 'am' // "1.01 Act : body..." inline or UPPERCASE headings
  }
};

const RUNTIME_CACHE_TTL_MS = 5 * 60 * 1000;

let db = null;
const statuses = {};
const outlineCaches = new Map();
let rescanTimer = null;

for (const book of Object.values(BOOKS)) {
  statuses[book.key] = { state: 'empty', lastError: null, chapters: 0, rules: 0, updatedAt: null, building: false, sourceMtime: 0 };
}

function bookPath(book) {
  return path.resolve(process.env.UPLOAD_DIR || path.join(__dirname, 'uploads'), book.filename);
}

const CHAPTER_WORDS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII'];

function chapterWordToNumber(word) {
  // OCR often turns "I" into "l" (small L); fold it back before matching.
  const normalized = String(word || '').trim().toUpperCase().replace(/L/g, 'I').replace(/\.$/, '');
  const idx = CHAPTER_WORDS.indexOf(normalized);
  return idx >= 0 ? idx + 1 : null;
}

function normalizeText(value) {
  return String(value || '')
    .replace(/\u00A0/g, ' ')
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Rebuild lines from pdf.js text items using y-coordinates. */
function itemsToLines(items) {
  const lines = [];
  let current = null;
  let lastY = null;
  for (const item of items) {
    const y = Math.round(item.transform[5]);
    if (current === null || Math.abs(y - lastY) > 3) {
      current = { y, parts: [] };
      lines.push(current);
      lastY = y;
    }
    current.parts.push(item.str);
    if (item.hasEOL) current.parts.push('\n');
  }
  return lines.map(line => normalizeText(line.parts.join(''))).filter(Boolean);
}

/** Clean running headers/footers and page numbers from body pages. */
function stripPageFurniture(lines) {
  const out = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    // Top of page: page number alone (i, ii, xviii, 1, 23, 108 ...)
    if (/^(?:[ivxlc]+|\d{1,4})$/i.test(line) && i < 2) continue;
    // Bare chapter title repeated as a running header
    if (i === 1 && line === out[0]) continue;
    if (/^(?:rule no|subject|page no)/i.test(line)) continue;
    // Bottom of page: correction-memo footer
    if (/\(correction memo no\.?\s*[\d.\/]+/i.test(line) && i > lines.length - 3) continue;
    out.push(line);
  }
  return out;
}

const isUpperish = value => {
  const letters = String(value || '').replace(/[^A-Za-z]/g, '');
  return letters.length > 0 && letters === letters.toUpperCase();
};

/**
 * Split the book into chapters and rules.
 * GRS headings look like:  GR.1.01 Short title and commencement. –
 * AM headings look like:   1.01 Act : Act means the Railways Act 1989...
 *                      or  2.01 SCOPE OF THE RULES (uppercase, may span lines)
 * Chapter markers:         CHAPTER I / CHAPTER l (OCR small L)
 */
function extractStructure(lines, book) {
  const chapters = [];
  const chapterByNumber = new Map();
  let currentChapter = null;
  let currentRule = null;
  // A stray "CHAPTER N" line (running header, TOC leftover) must never create
  // a second chapter block — reuse the existing one instead.
  const getOrCreateChapter = (num, title) => {
    if (chapterByNumber.has(num)) {
      const existing = chapterByNumber.get(num);
      if (title && (!existing.title || existing.title === `Chapter ${num}`)) existing.title = title;
      return existing;
    }
    const chapter = { number: num, title: title || `Chapter ${num}`, rules: [] };
    chapterByNumber.set(num, chapter);
    chapters.push(chapter);
    return chapter;
  };

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) continue;

    // Chapter heading. Case-sensitive on the word CHAPTER: real chapter
    // headings are ALL CAPS, while mixed-case "Chapter IX." lines are
    // running headers (page 50 of the Accident Manual has one).
    const chapterMatch = line.match(/^CHAPTER\s+([A-Za-z]+)\.?$/);
    if (chapterMatch) {
      const num = chapterWordToNumber(chapterMatch[1]);
      if (num) {
        // A "CHAPTER N" line repeating the current chapter is a running
        // header at a page top — skip it without breaking the open rule.
        if (currentChapter && currentChapter.number === num) continue;
        currentRule = null;
        // The chapter title follows the marker and may wrap onto further
        // uppercase lines ("...DEATH AND INJURY TO" + "PASSENGERS").
        let title = '';
        let j = i + 1;
        while (j < lines.length && j - i <= 3) {
          const candidate = lines[j];
          if (!candidate || /^CHAPTER\s/.test(candidate) || /^\d{1,3}$/.test(candidate)) break;
          if (book.titleStyle === 'am') {
            if (!isUpperish(candidate)) break;
            if (/^(?:[GR]{0,3}\d{1,2}\.\d{1,2})\b/.test(candidate) || /^[A-Z]\.\s/.test(candidate) || /^\([ivx]+\)/.test(candidate)) break;
          } else if (/^chapter\b/i.test(candidate)) {
            break;
          }
          title = title ? `${title} ${candidate}` : candidate;
          j += 1;
          if (book.titleStyle !== 'am') break; // GRS chapter titles are one line
        }
        currentChapter = getOrCreateChapter(num, normalizeText(title));
        i = j - 1;
        continue;
      }
    }

    // Rule heading: "GR.4.01 Limits of speed generally. –", "2.01. Title: -"
    // or AM's "1.01 Act : ..." / "2.01 SCOPE OF THE RULES" — the bold title
    // often sits on its own baseline, so also accept a bare number line.
    const ruleMatch = line.match(/^(?:GR\.)?(\d{1,2})\.(\d{1,2})\.?(?:\s+(.+))?$/i);
    if (ruleMatch && Number(ruleMatch[1]) >= 1 && Number(ruleMatch[1]) <= book.maxChapter) {
      const chapter = Number(ruleMatch[1]);
      const number = `${ruleMatch[1]}.${ruleMatch[2]}`;
      // Only accept headings inside the chapter the CHAPTER marker opened —
      // a "rule 5.10" cross-reference inside chapter 4 must not create
      // chapter 5 early, and appendix numbering (11.00–30.00 in the
      // Accident Manual) must not collide with chapter numbers.
      // Chapter switches happen via CHAPTER markers only.
      if (currentChapter && currentChapter.number === chapter && !currentChapter.rules.some(existing => existing.number === number)) {
        const last = currentChapter.rules[currentChapter.rules.length - 1];
        const rulePart = Number(ruleMatch[2]);
        const lastPart = last ? Number(last.number.split('.').pop()) : 0;
        // Rule numbers increase within a chapter; anything smaller is body text.
        if (rulePart > lastPart) {
          currentRule = {
            number: number,
            chapter: chapter,
            title: '',
            paragraphs: []
          };
          let restIndex = i;
          if (book.titleStyle === 'am') {
            // Accident Manual: title either sits before a colon on the same
            // line ("1.01 Act : means ...") or is an uppercase heading that
            // can continue onto the next line(s).
            const rest = normalizeText(ruleMatch[3] || '');
            if (rest && !isUpperish(rest) && rest.includes(':')) {
              const colonAt = rest.indexOf(':');
              currentRule.title = normalizeText(rest.slice(0, colonAt)).replace(/[\s.:\u2013\u2014-]+$/, '');
              const inlineBody = normalizeText(rest.slice(colonAt + 1));
              if (inlineBody) currentRule.paragraphs.push(inlineBody);
            } else if (rest) {
              currentRule.title = rest;
              let j = i + 1;
              while (
                currentRule.title && j < lines.length && j - i <= 3 &&
                isUpperish(lines[j]) && !/^chapter\b/i.test(lines[j]) &&
                !/^(?:GR\.)?\d{1,2}\.\d{1,2}\b/.test(lines[j]) && !/^\(\d+\)/.test(lines[j])
              ) {
                currentRule.title = `${currentRule.title} ${lines[j]}`;
                j += 1;
              }
              currentRule.title = normalizeText(currentRule.title).replace(/[\s:.\u2013\u2014-]+$/, '');
              restIndex = j - 1;
            }
          } else {
            currentRule.title = normalizeText(ruleMatch[3] || '').replace(/[\s.:\u2013\u2014-]*$/, '');
            if (!currentRule.title) {
              const nextLine = lines[i + 1];
              if (nextLine && !/^chapter\b/i.test(nextLine)) {
                currentRule.title = normalizeText(nextLine);
                restIndex = i + 1;
              }
            }
          }
          currentChapter.rules.push(currentRule);
          i = restIndex;
          continue;
        }
      }
    }

    if (currentRule) currentRule.paragraphs.push(line);
  }

  // GRS merges multi-line rule titles: lines before the first numbered
  // paragraph belong to the title. AM titles are already fully extracted.
  if (book.titleStyle === 'grs') {
    for (const chapter of chapters) {
      for (const rule of chapter.rules) {
        const bodyStart = rule.paragraphs.findIndex(p => /^\(\d+\)/.test(p) || /^[A-Z]\.\s/.test(p) || /^[a-z]/.test(p));
        if (rule.paragraphs.length && bodyStart > 0) {
          rule.title = normalizeText(`${rule.title} ${rule.paragraphs.slice(0, bodyStart).join(' ')}`);
          rule.paragraphs = rule.paragraphs.slice(bodyStart);
        }
      }
    }
  }
  for (const chapter of chapters) {
    for (const rule of chapter.rules) {
      rule.text = rule.paragraphs.join('\n\n');
      delete rule.paragraphs;
    }
  }
  return chapters;
}

function init(database) {
  db = database;
  db.exec(`
    CREATE TABLE IF NOT EXISTS rulebooks_chapters (
      book TEXT NOT NULL,
      number INTEGER NOT NULL,
      title TEXT NOT NULL,
      PRIMARY KEY (book, number)
    );
    CREATE TABLE IF NOT EXISTS rulebooks_rules (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      book TEXT NOT NULL,
      chapter INTEGER NOT NULL,
      number TEXT NOT NULL,
      title TEXT NOT NULL,
      body TEXT NOT NULL,
      UNIQUE(book, chapter, number)
    );
    CREATE TABLE IF NOT EXISTS rulebooks_edits (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      book TEXT NOT NULL,
      chapter INTEGER NOT NULL,
      number TEXT NOT NULL,
      title TEXT NOT NULL,
      body TEXT NOT NULL,
      orig_title TEXT,
      orig_body TEXT,
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(book, chapter, number)
    );
  `);
  db.exec('CREATE INDEX IF NOT EXISTS idx_rulebooks_rules_book_chapter ON rulebooks_rules(book, chapter)');
  try {
    db.exec("CREATE VIRTUAL TABLE IF NOT EXISTS rulebooks_fts USING fts5(book, chapter, number, title, body, content='rulebooks_rules', content_rowid='id')");
    db.exec(`CREATE TRIGGER IF NOT EXISTS rulebooks_rules_ai AFTER INSERT ON rulebooks_rules BEGIN
      INSERT INTO rulebooks_fts(rowid, book, chapter, number, title, body) VALUES (new.id, new.book, new.chapter, new.number, new.title, new.body);
    END`);
    db.exec(`CREATE TRIGGER IF NOT EXISTS rulebooks_rules_ad AFTER DELETE ON rulebooks_rules BEGIN
      INSERT INTO rulebooks_fts(rulebooks_fts, rowid, book, chapter, number, title, body) VALUES ('delete', old.id, old.book, old.chapter, old.number, old.title, old.body);
    END`);
    db.exec(`CREATE TRIGGER IF NOT EXISTS rulebooks_rules_au AFTER UPDATE ON rulebooks_rules BEGIN
      INSERT INTO rulebooks_fts(rulebooks_fts, rowid, book, chapter, number, title, body) VALUES ('delete', old.id, old.book, old.chapter, old.number, old.title, old.body);
      INSERT INTO rulebooks_fts(rowid, book, chapter, number, title, body) VALUES (new.id, new.book, new.chapter, new.number, new.title, new.body);
    END`);
  } catch (error) {
    console.error('Rule books full-text index unavailable:', error.message);
  }
  migrateLegacyGrs();

  for (const book of Object.values(BOOKS)) {
    if (fs.existsSync(bookPath(book))) {
      setTimeout(() => buildIndex(book), 3000);
    } else {
      console.log(`Rule books: ${book.filename} not found in the uploads folder, the ${book.label} reader stays empty.`);
    }
  }
  rescanTimer = setInterval(() => {
    for (const book of Object.values(BOOKS)) {
      const target = bookPath(book);
      if (fs.existsSync(target)) {
        const stat = fs.statSync(target);
        if (stat.mtimeMs !== (statuses[book.key].sourceMtime || 0)) buildIndex(book);
      }
    }
  }, 60000);
}

/** Copies data from the pre-multi-book grs_* tables once, then leaves them alone. */
function migrateLegacyGrs() {
  const hasLegacy = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='grs_chapters'").get();
  if (!hasLegacy) return;
  const alreadyMigrated = db.prepare('SELECT COUNT(*) c FROM rulebooks_chapters WHERE book = ?').get('grs').c > 0;
  if (alreadyMigrated) return;
  db.exec(`INSERT OR IGNORE INTO rulebooks_chapters (book, number, title) SELECT 'grs', number, title FROM grs_chapters`);
  db.exec(`INSERT OR IGNORE INTO rulebooks_rules (book, chapter, number, title, body) SELECT 'grs', chapter, number, title, body FROM grs_rules`);
  const hasLegacyEdits = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='grs_edits'").get();
  if (hasLegacyEdits) {
    db.exec(`INSERT OR IGNORE INTO rulebooks_edits (book, chapter, number, title, body, orig_title, orig_body, updated_at)
      SELECT 'grs', chapter, number, title, body, orig_title, orig_body, updated_at FROM grs_edits`);
  }
  console.log('Rule books: migrated the legacy grs_ tables into the multi-book store.');
}

function stop() {
  if (rescanTimer) clearInterval(rescanTimer);
}

function markStats(book) {
  const chapterRow = db.prepare('SELECT COUNT(*) c FROM rulebooks_chapters WHERE book = ?').get(book.key);
  const ruleRow = db.prepare('SELECT COUNT(*) c FROM rulebooks_rules WHERE book = ?').get(book.key);
  const status = statuses[book.key];
  status.chapters = chapterRow.c;
  status.rules = ruleRow.c;
}

function buildIndex(book) {
  const status = statuses[book.key];
  if (status.building) return;
  status.building = true;
  const target = bookPath(book);
  status.sourceMtime = fs.statSync(target).mtimeMs;
  console.log(`Rule books: parsing ${book.filename}...`);
  setImmediate(async () => {
    try {
      const { openPdfDocument, teardownPdf } = getRag();
      const loadingTask = await openPdfDocument(target);
      const pdf = await loadingTask.promise;
      const allLines = [];
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const content = await page.getTextContent();
        if (pageNumber > book.tocPages) allLines.push(...stripPageFurniture(itemsToLines(content.items)));
        if (pageNumber % 50 === 0) console.log(`Rule books: ${book.filename} page ${pageNumber}/${pdf.numPages}`);
        if (pageNumber % 10 === 0) await new Promise(resolve => setImmediate(resolve));
      }
      await teardownPdf(loadingTask, pdf);

      const chapters = extractStructure(allLines, book);
      if (!chapters.length) throw new Error('no chapters detected — heading format changed?');

      const write = db.transaction(() => {
        db.prepare('DELETE FROM rulebooks_rules WHERE book = ?').run(book.key);
        db.prepare('DELETE FROM rulebooks_chapters WHERE book = ?').run(book.key);
        const insChapter = db.prepare('INSERT INTO rulebooks_chapters (book, number, title) VALUES (?, ?, ?)');
        const insRule = db.prepare('INSERT OR REPLACE INTO rulebooks_rules (book, chapter, number, title, body) VALUES (?, ?, ?, ?, ?)');
        for (const chapter of chapters) {
          insChapter.run(book.key, chapter.number, chapter.title);
          for (const rule of chapter.rules) {
            insRule.run(book.key, chapter.number, rule.number, rule.title, rule.text);
          }
        }
        // Manual corrections from the admin editor are reapplied on top, so a
        // fresh parse of a new correction memo never wipes them.
        applyEditsToRules(book.key);
      });
      write();

      markStats(book);
      status.state = 'ready';
      status.lastError = null;
      status.updatedAt = new Date().toISOString();
      outlineCaches.delete(book.key);
      console.log(`Rule books: ${book.label} indexed ${status.chapters} chapters, ${status.rules} rules.`);
    } catch (error) {
      status.state = status.chapters > 0 ? 'ready' : 'error';
      status.lastError = error.message;
      console.error(`${book.label} indexing failed:`, error.message);
    } finally {
      status.building = false;
    }
  });
}

// ---- Admin corrections overlay ----
/** Reapplies every stored correction of a book to the live rule table. */
function applyEditsToRules(bookKey) {
  const edits = bookKey
    ? db.prepare('SELECT * FROM rulebooks_edits WHERE book = ? ORDER BY id').all(bookKey)
    : db.prepare('SELECT * FROM rulebooks_edits ORDER BY id').all();
  const update = db.prepare('UPDATE rulebooks_rules SET title = ?, body = ? WHERE book = ? AND chapter = ? AND number = ?');
  const insert = db.prepare('INSERT INTO rulebooks_rules (book, chapter, number, title, body) VALUES (?, ?, ?, ?, ?)');
  for (const edit of edits) {
    const result = update.run(edit.title, edit.body, edit.book, edit.chapter, edit.number);
    if (result.changes === 0) insert.run(edit.book, edit.chapter, edit.number, edit.title, edit.body);
  }
  return edits.length;
}

function getEdits(book) {
  if (book && BOOKS[book]) {
    return db.prepare(`
      SELECT e.id, e.book, e.chapter, e.number, e.title, e.body, e.orig_title, e.orig_body, e.updated_at,
        (SELECT COUNT(*) FROM rulebooks_rules r WHERE r.book = e.book AND r.chapter = e.chapter AND r.number = e.number) AS rule_exists
      FROM rulebooks_edits e WHERE e.book = ? ORDER BY e.book, e.chapter, e.number
    `).all(book);
  }
  return db.prepare(`
    SELECT e.id, e.book, e.chapter, e.number, e.title, e.body, e.orig_title, e.orig_body, e.updated_at,
      (SELECT COUNT(*) FROM rulebooks_rules r WHERE r.book = e.book AND r.chapter = e.chapter AND r.number = e.number) AS rule_exists
    FROM rulebooks_edits e ORDER BY e.book, e.chapter, e.number
  `).all();
}

function setEdit({ book, chapter, number, title, body }) {
  const bookConfig = BOOKS[book];
  if (!bookConfig) return { ok: false, error: 'Unknown manual' };
  const chapterNum = Number(chapter);
  const ruleNumber = String(number || '').trim();
  const cleanTitle = String(title || '').trim();
  const cleanBody = String(body || '').trim();
  if (!Number.isInteger(chapterNum) || chapterNum < 1 || chapterNum > bookConfig.maxChapter) {
    return { ok: false, error: `Chapter must be between 1 and ${bookConfig.maxChapter}` };
  }
  if (!/^\d{1,2}\.\d{1,2}$/.test(ruleNumber)) return { ok: false, error: 'Rule number must look like 4.01' };
  if (!cleanTitle) return { ok: false, error: 'Title is required' };
  if (!cleanBody) return { ok: false, error: 'Text is required' };
  const existing = db.prepare('SELECT * FROM rulebooks_edits WHERE book = ? AND chapter = ? AND number = ?').get(book, chapterNum, ruleNumber);
  let origTitle = null;
  let origBody = null;
  if (existing) {
    origTitle = existing.orig_title;
    origBody = existing.orig_body;
  } else {
    const rule = db.prepare('SELECT title, body FROM rulebooks_rules WHERE book = ? AND chapter = ? AND number = ?').get(book, chapterNum, ruleNumber);
    origTitle = rule ? rule.title : null;
    origBody = rule ? rule.body : null;
  }
  db.prepare(`
    INSERT INTO rulebooks_edits (book, chapter, number, title, body, orig_title, orig_body, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    ON CONFLICT(book, chapter, number) DO UPDATE SET title = excluded.title, body = excluded.body, updated_at = excluded.updated_at
  `).run(book, chapterNum, ruleNumber, cleanTitle, cleanBody, origTitle, origBody);
  const updated = db.prepare('UPDATE rulebooks_rules SET title = ?, body = ? WHERE book = ? AND chapter = ? AND number = ?')
    .run(cleanTitle, cleanBody, book, chapterNum, ruleNumber);
  if (updated.changes === 0) {
    // The rule was not in the parsed book — add it (admin-added rule).
    db.prepare('INSERT INTO rulebooks_rules (book, chapter, number, title, body) VALUES (?, ?, ?, ?, ?)')
      .run(book, chapterNum, ruleNumber, cleanTitle, cleanBody);
  }
  outlineCaches.delete(book);
  return { ok: true };
}

/** Removes a correction and restores the original parsed text (or drops an added rule). */
function clearEdit(book, chapter, number) {
  const bookConfig = BOOKS[book];
  if (!bookConfig) return { ok: false, error: 'Unknown manual' };
  const chapterNum = Number(chapter);
  const ruleNumber = String(number || '').trim();
  const edit = db.prepare('SELECT * FROM rulebooks_edits WHERE book = ? AND chapter = ? AND number = ?').get(book, chapterNum, ruleNumber);
  if (!edit) return { ok: false, error: 'No correction stored for this rule' };
  if (edit.orig_title === null) {
    db.prepare('DELETE FROM rulebooks_rules WHERE book = ? AND chapter = ? AND number = ?').run(book, chapterNum, ruleNumber);
  } else {
    db.prepare('UPDATE rulebooks_rules SET title = ?, body = ? WHERE book = ? AND chapter = ? AND number = ?')
      .run(edit.orig_title, edit.orig_body, book, chapterNum, ruleNumber);
  }
  db.prepare('DELETE FROM rulebooks_edits WHERE id = ?').run(edit.id);
  outlineCaches.delete(book);
  return { ok: true };
}

function getStatus(book) {
  if (book && BOOKS[book]) {
    const { sourceMtime, ...rest } = statuses[book];
    return rest;
  }
  const all = {};
  for (const key of Object.keys(BOOKS)) {
    const { sourceMtime, ...rest } = statuses[key];
    all[key] = rest;
  }
  return all;
}

function getOutline(book) {
  if (!BOOKS[book]) return [];
  const now = Date.now();
  const cached = outlineCaches.get(book);
  if (cached && now - cached.at < RUNTIME_CACHE_TTL_MS) return cached.outline;
  const rows = db.prepare(`
    SELECT c.number, c.title, r.number AS rule_number, r.title AS rule_title
    FROM rulebooks_chapters c LEFT JOIN rulebooks_rules r ON r.book = c.book AND r.chapter = c.number
    WHERE c.book = ?
    ORDER BY c.number, r.id
  `).all(book);
  const map = new Map();
  for (const row of rows) {
    if (!map.has(row.number)) map.set(row.number, { number: row.number, title: row.title, rules: [] });
    if (row.rule_number) map.get(row.number).rules.push({ number: row.rule_number, title: row.rule_title });
  }
  const ruleOrder = (a, b) => {
    const [ac, ar] = a.number.split('.').map(Number);
    const [bc, br] = b.number.split('.').map(Number);
    return ac - bc || ar - br;
  };
  const outline = Array.from(map.values()).map(chapter => ({
    number: chapter.number,
    title: chapter.title,
    rules: chapter.rules.slice().sort(ruleOrder)
  }));
  outlineCaches.set(book, { at: now, outline });
  return outline;
}

function getSection(book, chapter, number) {
  if (!BOOKS[book]) return undefined;
  return db.prepare('SELECT id, book, chapter, number, title, body FROM rulebooks_rules WHERE book = ? AND chapter = ? AND number = ?')
    .get(book, Number(chapter), String(number || '').trim());
}

function search(query, book, limit = 40) {
  const cleaned = normalizeText(query).replace(/["*()]/g, ' ').trim();
  if (!cleaned) return { results: [], total: 0, terms: [] };
  const terms = cleaned.split(/\s+/).filter(Boolean).slice(0, 8);
  const ftsQuery = terms.map(term => `"${term}"`).join(' ');
  const bookFilter = BOOKS[book] ? book : null;
  const cappedLimit = Math.min(Math.max(Number(limit) || 40, 1), 100);
  try {
    const rows = bookFilter
      ? db.prepare(`
          SELECT r.id, r.book, r.chapter, r.number, r.title, snippet(rulebooks_fts, 4, '[[', ']]', '…', 18) AS snippet,
                 bm25(rulebooks_fts) AS rank
          FROM rulebooks_fts f JOIN rulebooks_rules r ON r.id = f.rowid
          WHERE rulebooks_fts MATCH ? AND r.book = ?
          ORDER BY rank LIMIT ?
        `).all(ftsQuery, bookFilter, cappedLimit)
      : db.prepare(`
          SELECT r.id, r.book, r.chapter, r.number, r.title, snippet(rulebooks_fts, 4, '[[', ']]', '…', 18) AS snippet,
                 bm25(rulebooks_fts) AS rank
          FROM rulebooks_fts f JOIN rulebooks_rules r ON r.id = f.rowid
          WHERE rulebooks_fts MATCH ?
          ORDER BY rank LIMIT ?
        `).all(ftsQuery, cappedLimit);
    return { results: rows, total: rows.length, terms };
  } catch (error) {
    console.error('Rule book search failed:', error.message);
    return { results: [], total: 0, terms };
  }
}

function reindex(book) {
  const targets = book && BOOKS[book] ? [BOOKS[book]] : Object.values(BOOKS);
  let started = 0;
  let error = null;
  for (const target of targets) {
    if (!fs.existsSync(bookPath(target))) {
      error = `${target.filename} is not in the uploads folder`;
      continue;
    }
    buildIndex(target);
    started += 1;
  }
  if (!started && error) return { ok: false, error };
  return { ok: true };
}

module.exports = { init, stop, getStatus, getOutline, getSection, search, reindex, getEdits, setEdit, clearEdit, BOOKS, GR_BOOK_FILENAME: BOOKS.grs.filename };
