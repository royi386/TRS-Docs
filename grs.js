/**
 * General Rules (GRS) structured reader.
 *
 * Reads the GRS Full.pdf text layer once, splits it into chapters and
 * individually addressable rules (1.01, 1.02, ...), stores them in SQLite
 * with a full-text index, and serves them to the reader page.
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

const GR_BOOK_FILENAME = 'GRS Full.pdf';
const BOOK_TOC_PAGES = 22; // front matter + arrangement of rules, no rule bodies
const RUNTIME_CACHE_TTL_MS = 5 * 60 * 1000;

let db = null;
let grsStatus = { state: 'empty', lastError: null, chapters: 0, rules: 0, updatedAt: null, building: false };
let rescanTimer = null;
let outlineCache = null;
let outlineCacheAt = 0;

const CHAPTER_WORDS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII'];

function chapterWordToNumber(word) {
  const idx = CHAPTER_WORDS.indexOf(String(word || '').trim().toUpperCase().replace(/\.$/, ''));
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
    // Top of page: roman page number alone (i, ii, xviii, 1, 23, 108 ...)
    if (/^(?:[ivxlc]+|\d{1,4})$/i.test(line) && i < 2) continue;
    // Running chapter header lines at the top of the page
    if (/^chapter\s+[ivx]+\b/i.test(line) && i < 2) continue;
    // Bare chapter title repeated as a running header
    if (i === 1 && line === out[0]) continue;
    if (/^(?:rule no|subject|page no)/i.test(line)) continue;
    // Bottom of page: correction-memo footer
    if (/\(correction memo no\.?\s*[\d.\/]+/i.test(line) && i > lines.length - 3) continue;
    out.push(line);
  }
  return out;
}

/**
 * Split the book into chapters and rules.
 * Headings look like:  GR.1.01 Short title and commencement. –
 * Chapter markers:     CHAPTER I / CHAPTER Il (OCR-ish capital i)
 */
function extractStructure(lines) {
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

  const pushRuleLine = (line) => {
    if (!currentRule) return;
    currentRule.paragraphs.push(line);
  };

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) continue;

    // Chapter heading (tolerates OCR-mixed "Il" for "II" via word position)
    const chapterMatch = line.match(/^chapter\s+([ivxlc]+)\b\.?$/i);
    if (chapterMatch) {
      const num = chapterWordToNumber(chapterMatch[1]);
      if (num) {
        const titleLine = lines[i + 1] && !/^chapter\b/i.test(lines[i + 1]) && !/^\d{1,3}$/.test(lines[i + 1]) ? lines[i + 1] : '';
        currentChapter = getOrCreateChapter(num, normalizeText(titleLine));
        currentRule = null;
        continue;
      }
    }

    // Rule heading: "GR.4.01 Limits of speed generally. –" or "2.01. Title: -"
    // — the bold title often sits on its own baseline, so also accept a bare
    // number line and take the following line as the title.
    const ruleMatch = line.match(/^(?:GR\.)?(\d{1,2})\.(\d{1,2})\.?(?:\s+(.+))?$/i);
    if (ruleMatch && Number(ruleMatch[1]) >= 1 && Number(ruleMatch[1]) <= 18) {
      const chapter = Number(ruleMatch[1]);
      const number = `${ruleMatch[1]}.${ruleMatch[2]}`;
      // Only accept headings inside the chapter the CHAPTER marker opened —
      // a "rule 5.10" cross-reference inside chapter 4 must not create
      // chapter 5 early. Chapter switches happen via CHAPTER markers only.
      if (currentChapter && currentChapter.number === chapter && !currentChapter.rules.some(existing => existing.number === number)) {
        const last = currentChapter.rules[currentChapter.rules.length - 1];
        const asNumber = value => Number(number.split('.').pop());
        const lastNumber = last ? Number(last.number.split('.').pop()) : 0;
        // Rule numbers increase within a chapter; anything smaller is body text.
        if (asNumber() > lastNumber) {
          let title = normalizeText(ruleMatch[3] || '');
          if (!title) {
            const nextLine = lines[i + 1];
            if (nextLine && !/^chapter\b/i.test(nextLine)) {
              title = normalizeText(nextLine);
              i += 1;
            }
          }
          currentRule = {
            number: number,
            chapter: chapter,
            title: title.replace(/[\s.:\u2013\u2014-]*$/, '').trim(),
            paragraphs: []
          };
          currentChapter.rules.push(currentRule);
          continue;
        }
      }
    }

    if (currentRule) pushRuleLine(line);
  }

  // Merge multi-line rule titles: lines before the first numbered paragraph
  for (const chapter of chapters) {
    for (const rule of chapter.rules) {
      const bodyStart = rule.paragraphs.findIndex(p => /^\(\d+\)/.test(p) || /^[A-Z]\.\s/.test(p) || /^[a-z]/.test(p));
      if (rule.paragraphs.length && bodyStart > 0) {
        rule.title = normalizeText(`${rule.title} ${rule.paragraphs.slice(0, bodyStart).join(' ')}`);
        rule.paragraphs = rule.paragraphs.slice(bodyStart);
      }
      rule.text = rule.paragraphs.join('\n\n');
      delete rule.paragraphs;
    }
  }
  return chapters;
}

function init(database) {
  db = database;
  db.exec(`
    CREATE TABLE IF NOT EXISTS grs_chapters (
      number INTEGER PRIMARY KEY,
      title TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS grs_rules (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      chapter INTEGER NOT NULL,
      number TEXT NOT NULL,
      title TEXT NOT NULL,
      body TEXT NOT NULL,
      UNIQUE(chapter, number)
    );
    CREATE TABLE IF NOT EXISTS grs_edits (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      chapter INTEGER NOT NULL,
      number TEXT NOT NULL,
      title TEXT NOT NULL,
      body TEXT NOT NULL,
      orig_title TEXT,
      orig_body TEXT,
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(chapter, number)
    );
  `);
  try {
    db.exec('CREATE INDEX IF NOT EXISTS idx_grs_rules_chapter ON grs_rules(chapter)');
    db.exec("CREATE VIRTUAL TABLE IF NOT EXISTS grs_rules_fts USING fts5(chapter, number, title, body, content='grs_rules', content_rowid='id')");
    db.exec(`CREATE TRIGGER IF NOT EXISTS grs_rules_ai AFTER INSERT ON grs_rules BEGIN
      INSERT INTO grs_rules_fts(rowid, chapter, number, title, body) VALUES (new.id, new.chapter, new.number, new.title, new.body);
    END`);
    db.exec(`CREATE TRIGGER IF NOT EXISTS grs_rules_ad AFTER DELETE ON grs_rules BEGIN
      INSERT INTO grs_rules_fts(grs_rules_fts, rowid, chapter, number, title, body) VALUES ('delete', old.id, old.chapter, old.number, old.title, old.body);
    END`);
    db.exec(`CREATE TRIGGER IF NOT EXISTS grs_rules_au AFTER UPDATE ON grs_rules BEGIN
      INSERT INTO grs_rules_fts(grs_rules_fts, rowid, chapter, number, title, body) VALUES ('delete', old.id, old.chapter, old.number, old.title, old.body);
      INSERT INTO grs_rules_fts(rowid, chapter, number, title, body) VALUES (new.id, new.chapter, new.number, new.title, new.body);
    END`);
  } catch (error) {
    console.error('GRS full-text index unavailable:', error.message);
  }

  const bookPath = path.resolve(process.env.UPLOAD_DIR || path.join(__dirname, 'uploads'), GR_BOOK_FILENAME);
  if (fs.existsSync(bookPath)) {
    setTimeout(() => buildIndex(bookPath), 3000);
  } else {
    console.log(`GRS reader: ${GR_BOOK_FILENAME} not found in the uploads folder, reader stays empty.`);
  }
  rescanTimer = setInterval(() => {
    const target = path.resolve(process.env.UPLOAD_DIR || path.join(__dirname, 'uploads'), GR_BOOK_FILENAME);
    if (fs.existsSync(target)) {
      const stat = fs.statSync(target);
      if (stat.mtimeMs !== (grsStatus.sourceMtime || 0)) buildIndex(target);
    }
  }, 60000);
}

function stop() {
  if (rescanTimer) clearInterval(rescanTimer);
}

function markStats() {
  const chapterRow = db.prepare('SELECT COUNT(*) c FROM grs_chapters').get();
  const ruleRow = db.prepare('SELECT COUNT(*) c FROM grs_rules').get();
  grsStatus.chapters = chapterRow.c;
  grsStatus.rules = ruleRow.c;
}

function buildIndex(bookPath) {
  if (grsStatus.building) return;
  grsStatus.building = true;
  grsStatus.sourceMtime = fs.statSync(bookPath).mtimeMs;
  console.log(`GRS reader: parsing ${path.basename(bookPath)}...`);
  setImmediate(async () => {
    try {
      const { openPdfDocument, teardownPdf } = getRag();
      const loadingTask = await openPdfDocument(bookPath);
      const pdf = await loadingTask.promise;
      const allLines = [];
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const content = await page.getTextContent();
        if (pageNumber > BOOK_TOC_PAGES) allLines.push(...stripPageFurniture(itemsToLines(content.items)));
        if (pageNumber % 50 === 0) console.log(`GRS reader: page ${pageNumber}/${pdf.numPages}`);
        if (pageNumber % 10 === 0) await new Promise(resolve => setImmediate(resolve));
      }
      await teardownPdf(loadingTask, pdf);

      const chapters = extractStructure(allLines);
      if (!chapters.length) throw new Error('no chapters detected — heading format changed?');

      const write = db.transaction(() => {
        db.exec('DELETE FROM grs_rules');
        db.exec('DELETE FROM grs_chapters');
        const insChapter = db.prepare('INSERT INTO grs_chapters (number, title) VALUES (?, ?)');
        const insRule = db.prepare('INSERT OR REPLACE INTO grs_rules (chapter, number, title, body) VALUES (?, ?, ?, ?)');
        for (const chapter of chapters) {
          insChapter.run(chapter.number, chapter.title);
          for (const rule of chapter.rules) {
            insRule.run(chapter.number, rule.number, rule.title, rule.text);
          }
        }
        // Manual corrections from the admin editor are reapplied on top, so a
        // fresh parse of a new correction memo never wipes them.
        applyEditsToRules();
      });
      write();

      markStats();
      grsStatus.state = 'ready';
      grsStatus.lastError = null;
      grsStatus.updatedAt = new Date().toISOString();
      outlineCache = null;
      console.log(`GRS reader: indexed ${grsStatus.chapters} chapters, ${grsStatus.rules} rules.`);
    } catch (error) {
      grsStatus.state = grsStatus.chapters > 0 ? 'ready' : 'error';
      grsStatus.lastError = error.message;
      console.error('GRS indexing failed:', error.message);
    } finally {
      grsStatus.building = false;
    }
  });
}

// ---- Admin corrections overlay ----
/** Reapplies every stored correction to the live rule table. */
function applyEditsToRules() {
  const edits = db.prepare('SELECT * FROM grs_edits ORDER BY id').all();
  const update = db.prepare('UPDATE grs_rules SET title = ?, body = ? WHERE chapter = ? AND number = ?');
  const insert = db.prepare('INSERT INTO grs_rules (chapter, number, title, body) VALUES (?, ?, ?, ?)');
  for (const edit of edits) {
    const result = update.run(edit.title, edit.body, edit.chapter, edit.number);
    if (result.changes === 0) insert.run(edit.chapter, edit.number, edit.title, edit.body);
  }
  return edits.length;
}

function getEdits() {
  return db.prepare(`
    SELECT e.id, e.chapter, e.number, e.title, e.body, e.orig_title, e.orig_body, e.updated_at,
      (SELECT COUNT(*) FROM grs_rules r WHERE r.chapter = e.chapter AND r.number = e.number) AS rule_exists
    FROM grs_edits e ORDER BY e.chapter, e.number
  `).all();
}

function setEdit({ chapter, number, title, body }) {
  const chapterNum = Number(chapter);
  const ruleNumber = String(number || '').trim();
  const cleanTitle = String(title || '').trim();
  const cleanBody = String(body || '').trim();
  if (!Number.isInteger(chapterNum) || chapterNum < 1 || chapterNum > 18) return { ok: false, error: 'Chapter must be between 1 and 18' };
  if (!/^\d{1,2}\.\d{1,2}$/.test(ruleNumber)) return { ok: false, error: 'Rule number must look like 4.01' };
  if (!cleanTitle) return { ok: false, error: 'Title is required' };
  if (!cleanBody) return { ok: false, error: 'Text is required' };
  const existing = db.prepare('SELECT * FROM grs_edits WHERE chapter = ? AND number = ?').get(chapterNum, ruleNumber);
  let origTitle = null;
  let origBody = null;
  if (existing) {
    origTitle = existing.orig_title;
    origBody = existing.orig_body;
  } else {
    const rule = db.prepare('SELECT title, body FROM grs_rules WHERE chapter = ? AND number = ?').get(chapterNum, ruleNumber);
    origTitle = rule ? rule.title : null;
    origBody = rule ? rule.body : null;
  }
  db.prepare(`
    INSERT INTO grs_edits (chapter, number, title, body, orig_title, orig_body, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    ON CONFLICT(chapter, number) DO UPDATE SET title = excluded.title, body = excluded.body, updated_at = excluded.updated_at
  `).run(chapterNum, ruleNumber, cleanTitle, cleanBody, origTitle, origBody);
  const updated = db.prepare('UPDATE grs_rules SET title = ?, body = ? WHERE chapter = ? AND number = ?')
    .run(cleanTitle, cleanBody, chapterNum, ruleNumber);
  if (updated.changes === 0) {
    // The rule was not in the parsed book — add it (admin-added rule).
    db.prepare('INSERT INTO grs_rules (chapter, number, title, body) VALUES (?, ?, ?, ?)')
      .run(chapterNum, ruleNumber, cleanTitle, cleanBody);
  }
  outlineCache = null;
  return { ok: true };
}

/** Removes a correction and restores the original parsed text (or drops an added rule). */
function clearEdit(chapter, number) {
  const chapterNum = Number(chapter);
  const ruleNumber = String(number || '').trim();
  const edit = db.prepare('SELECT * FROM grs_edits WHERE chapter = ? AND number = ?').get(chapterNum, ruleNumber);
  if (!edit) return { ok: false, error: 'No correction stored for this rule' };
  if (edit.orig_title === null) {
    db.prepare('DELETE FROM grs_rules WHERE chapter = ? AND number = ?').run(chapterNum, ruleNumber);
  } else {
    db.prepare('UPDATE grs_rules SET title = ?, body = ? WHERE chapter = ? AND number = ?')
      .run(edit.orig_title, edit.orig_body, chapterNum, ruleNumber);
  }
  db.prepare('DELETE FROM grs_edits WHERE id = ?').run(edit.id);
  outlineCache = null;
  return { ok: true };
}

function getStatus() {
  return {
    state: grsStatus.state,
    building: grsStatus.building,
    chapters: grsStatus.chapters,
    rules: grsStatus.rules,
    lastError: grsStatus.lastError,
    updatedAt: grsStatus.updatedAt
  };
}

function getOutline() {
  const now = Date.now();
  if (outlineCache && now - outlineCacheAt < RUNTIME_CACHE_TTL_MS) return outlineCache;
  const chapters = db.prepare(`
    SELECT c.number, c.title, r.number AS rule_number, r.title AS rule_title
    FROM grs_chapters c LEFT JOIN grs_rules r ON r.chapter = c.number
    ORDER BY c.number, r.id
  `).all();
  const map = new Map();
  for (const row of chapters) {
    if (!map.has(row.number)) map.set(row.number, { number: row.number, title: row.title, rules: [] });
    if (row.rule_number) map.get(row.number).rules.push({ number: row.rule_number, title: row.rule_title });
  }
  const ruleOrder = (a, b) => {
    const [ac, ar] = a.number.split('.').map(Number);
    const [bc, br] = b.number.split('.').map(Number);
    return ac - bc || ar - br;
  };
  outlineCache = Array.from(map.values()).map(chapter => ({
    number: chapter.number,
    title: chapter.title,
    rules: chapter.rules.slice().sort(ruleOrder)
  }));
  outlineCacheAt = now;
  return outlineCache;
}

function getSection(chapter, number) {
  return db.prepare('SELECT id, chapter, number, title, body FROM grs_rules WHERE chapter = ? AND number = ?')
    .get(Number(chapter), String(number || '').trim());
}

function search(query, limit = 40) {
  const cleaned = normalizeText(query).replace(/["*()]/g, ' ').trim();
  if (!cleaned) return { results: [], total: 0, terms: [] };
  const terms = cleaned.split(/\s+/).filter(Boolean).slice(0, 8);
  const ftsQuery = terms.map(term => `"${term}"`).join(' ');
  try {
    const rows = db.prepare(`
      SELECT r.id, r.chapter, r.number, r.title, snippet(grs_rules_fts, 3, '[[', ']]', '…', 18) AS snippet,
             bm25(grs_rules_fts) AS rank
      FROM grs_rules_fts f JOIN grs_rules r ON r.id = f.rowid
      WHERE grs_rules_fts MATCH ?
      ORDER BY rank LIMIT ?
    `).all(ftsQuery, Math.min(Math.max(Number(limit) || 40, 1), 100));
    return { results: rows, total: rows.length, terms };
  } catch (error) {
    console.error('GRS search failed:', error.message);
    return { results: [], total: 0, terms };
  }
}

function reindex() {
  const target = path.resolve(process.env.UPLOAD_DIR || path.join(__dirname, 'uploads'), GR_BOOK_FILENAME);
  if (!fs.existsSync(target)) return { ok: false, error: `${GR_BOOK_FILENAME} is not in the uploads folder` };
  buildIndex(target);
  return { ok: true };
}

module.exports = { init, stop, getStatus, getOutline, getSection, search, reindex, getEdits, setEdit, clearEdit, GR_BOOK_FILENAME };
