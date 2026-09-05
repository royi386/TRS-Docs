const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const Database = require('better-sqlite3');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD;
const adminTokens = new Set();
const drawingTokens = new Map();
const uploadDirectory = process.env.UPLOAD_DIR || path.join(__dirname, 'uploads');
const feedbackUploadDirectory = process.env.FEEDBACK_UPLOAD_DIR || path.join(__dirname, 'feedback-uploads');
const databasePath = process.env.DB_PATH || path.join(__dirname, 'smi_tc.db');
fs.mkdirSync(uploadDirectory, { recursive: true });
fs.mkdirSync(feedbackUploadDirectory, { recursive: true });

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Database setup
const db = new Database(databasePath);

// Create tables
db.exec(`
  CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    original_name TEXT NOT NULL,
    caption TEXT NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    mime_type TEXT
  );
  
  CREATE INDEX IF NOT EXISTS idx_caption ON documents(caption);
  CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename);
`);

try {
  db.exec('ALTER TABLE documents ADD COLUMN document_number TEXT NOT NULL DEFAULT \'\'');
} catch (error) {
  if (!error.message.includes('duplicate column name')) throw error;
}
db.exec('CREATE INDEX IF NOT EXISTS idx_document_number ON documents(document_number)');
try {
  db.exec('ALTER TABLE documents ADD COLUMN document_type TEXT NOT NULL DEFAULT \'\'');
} catch (error) {
  if (!error.message.includes('duplicate column name')) throw error;
}
db.exec(`
  UPDATE documents
  SET document_type = CASE
    WHEN document_number LIKE 'SMI%' THEN 'SMI'
    WHEN document_number LIKE 'MS%' THEN 'MS'
    WHEN document_number LIKE 'TC%' THEN 'TC'
    WHEN document_number LIKE 'Drawings%' THEN 'Drawings'
    ELSE document_type
  END
  WHERE document_type = ''
`);
db.exec('CREATE INDEX IF NOT EXISTS idx_document_type ON documents(document_type)');
try {
  db.exec('ALTER TABLE documents ADD COLUMN keywords TEXT NOT NULL DEFAULT \'\'');
} catch (error) {
  if (!error.message.includes('duplicate column name')) throw error;
}
db.exec('CREATE INDEX IF NOT EXISTS idx_keywords ON documents(keywords)');
db.exec(`
  CREATE TABLE IF NOT EXISTS document_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
    requires_login INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);
let addedDocumentTypeLoginColumn = false;
try {
  db.exec('ALTER TABLE document_types ADD COLUMN requires_login INTEGER NOT NULL DEFAULT 0');
  addedDocumentTypeLoginColumn = true;
} catch (error) {
  if (!error.message.includes('duplicate column name')) throw error;
}
const addDocumentType = db.prepare('INSERT OR IGNORE INTO document_types (name, requires_login) VALUES (?, ?)');
['SMI', 'MS', 'TC', 'Drawings'].forEach(name => addDocumentType.run(name, name === 'Drawings' ? 1 : 0));
if (addedDocumentTypeLoginColumn) db.prepare("UPDATE document_types SET requires_login = 1 WHERE name = 'Drawings'").run();
db.prepare(`
  INSERT OR IGNORE INTO document_types (name)
  SELECT DISTINCT TRIM(document_type) FROM documents WHERE TRIM(document_type) <> ''
`).run();
db.exec(`
  CREATE TABLE IF NOT EXISTS feedback_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_name TEXT NOT NULL DEFAULT '',
    sender_contact TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    original_name TEXT NOT NULL DEFAULT '',
    file_size INTEGER,
    mime_type TEXT,
    submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE INDEX IF NOT EXISTS idx_feedback_submitted_at ON feedback_submissions(submitted_at);
`);
db.exec(`
  CREATE TABLE IF NOT EXISTS drawing_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL DEFAULT '',
    designation TEXT NOT NULL DEFAULT '',
    division TEXT NOT NULL DEFAULT '',
    phone_number TEXT NOT NULL DEFAULT '',
    account_status TEXT NOT NULL DEFAULT 'approved',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);
[
  ['full_name', "TEXT NOT NULL DEFAULT ''"],
  ['designation', "TEXT NOT NULL DEFAULT ''"],
  ['division', "TEXT NOT NULL DEFAULT ''"],
  ['phone_number', "TEXT NOT NULL DEFAULT ''"],
  ['account_status', "TEXT NOT NULL DEFAULT 'approved'"]
].forEach(([column, definition]) => {
  try {
    db.exec(`ALTER TABLE drawing_users ADD COLUMN ${column} ${definition}`);
  } catch (error) {
    if (!error.message.includes('duplicate column name')) throw error;
  }
});

function validateDrawingUser(username, password, passwordRequired = true) {
  if (!username || username.length < 3 || username.length > 80) return 'Username must be between 3 and 80 characters';
  if (passwordRequired && (!password || password.length < 8 || password.length > 200)) return 'Password must be between 8 and 200 characters';
  return '';
}

function validateDrawingRegistration({ username, password, fullName, designation, division, phoneNumber }) {
  const credentialError = validateDrawingUser(username, password);
  if (credentialError) return credentialError;
  if (!fullName || fullName.length > 120) return 'Name is required and must be 120 characters or fewer';
  if (!designation || designation.length > 120) return 'Designation is required and must be 120 characters or fewer';
  if (!division || division.length > 120) return 'Division is required and must be 120 characters or fewer';
  if (!phoneNumber || phoneNumber.length < 6 || phoneNumber.length > 30) return 'Enter a valid phone number';
  return '';
}

function hashPassword(password) {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return `${salt}:${hash}`;
}

function verifyPassword(password, passwordHash) {
  const [salt, savedHash] = String(passwordHash).split(':');
  if (!salt || !savedHash) return false;
  const suppliedHash = crypto.scryptSync(password, salt, 64).toString('hex');
  return crypto.timingSafeEqual(Buffer.from(suppliedHash, 'hex'), Buffer.from(savedHash, 'hex'));
}

function revokeDrawingTokens(userId) {
  for (const [token, tokenUserId] of drawingTokens) if (tokenUserId === Number(userId)) drawingTokens.delete(token);
}

function isAdmin(req) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  return Boolean(token && adminTokens.has(token));
}

function isDrawingUser(req) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  return Boolean(token && drawingTokens.has(token));
}

function requireAdmin(req, res, next) {
  if (!ADMIN_PASSWORD) {
    return res.status(503).json({ error: 'Admin access is not configured on the server' });
  }
  if (!isAdmin(req)) return res.status(401).json({ error: 'Admin login required' });
  next();
}

// Multer storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDirectory);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, 'tc-' + uniqueSuffix + ext);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed!'), false);
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

const feedbackUpload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, feedbackUploadDirectory),
    filename: (req, file, cb) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
      cb(null, 'feedback-' + uniqueSuffix + path.extname(file.originalname));
    }
  }),
  limits: { fileSize: 25 * 1024 * 1024 }
});

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/api/admin/login', (req, res) => {
  if (!ADMIN_PASSWORD) return res.status(503).json({ error: 'Admin access is not configured on the server' });
  const suppliedPassword = Buffer.from(typeof req.body.password === 'string' ? req.body.password : '');
  const expectedPassword = Buffer.from(ADMIN_PASSWORD);
  if (suppliedPassword.length !== expectedPassword.length || !crypto.timingSafeEqual(suppliedPassword, expectedPassword)) {
    return res.status(401).json({ error: 'Incorrect password' });
  }
  const token = crypto.randomBytes(32).toString('hex');
  adminTokens.add(token);
  res.json({ token });
});

app.post('/api/admin/logout', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (token) adminTokens.delete(token);
  res.json({ success: true });
});

app.get('/api/drawing-users', requireAdmin, (req, res) => {
  res.json(db.prepare('SELECT id, username, full_name, designation, division, phone_number, account_status, created_at FROM drawing_users ORDER BY CASE account_status WHEN \'pending\' THEN 0 ELSE 1 END, username COLLATE NOCASE').all());
});

app.post('/api/drawing-users', requireAdmin, (req, res) => {
  const username = typeof req.body.username === 'string' ? req.body.username.trim() : '';
  const password = typeof req.body.password === 'string' ? req.body.password : '';
  const validationError = validateDrawingUser(username, password);
  if (validationError) return res.status(400).json({ error: validationError });
  try {
    const result = db.prepare("INSERT INTO drawing_users (username, password_hash, account_status) VALUES (?, ?, 'approved')").run(username, hashPassword(password));
    res.status(201).json({ id: result.lastInsertRowid, username });
  } catch (error) {
    if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') return res.status(409).json({ error: 'This username already exists' });
    console.error('Create drawing user error:', error);
    res.status(500).json({ error: 'Unable to create user' });
  }
});

app.post('/api/drawings/register', (req, res) => {
  const registration = {
    username: typeof req.body.username === 'string' ? req.body.username.trim() : '',
    password: typeof req.body.password === 'string' ? req.body.password : '',
    fullName: typeof req.body.fullName === 'string' ? req.body.fullName.trim() : '',
    designation: typeof req.body.designation === 'string' ? req.body.designation.trim() : '',
    division: typeof req.body.division === 'string' ? req.body.division.trim() : '',
    phoneNumber: typeof req.body.phoneNumber === 'string' ? req.body.phoneNumber.trim() : ''
  };
  const validationError = validateDrawingRegistration(registration);
  if (validationError) return res.status(400).json({ error: validationError });
  try {
    db.prepare(`
      INSERT INTO drawing_users (username, password_hash, full_name, designation, division, phone_number, account_status)
      VALUES (?, ?, ?, ?, ?, ?, 'pending')
    `).run(registration.username, hashPassword(registration.password), registration.fullName, registration.designation, registration.division, registration.phoneNumber);
    res.status(201).json({ success: true, message: 'Your account request has been sent for approval' });
  } catch (error) {
    if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') return res.status(409).json({ error: 'This username is already registered' });
    console.error('Register drawing user error:', error);
    res.status(500).json({ error: 'Unable to submit account request' });
  }
});

app.patch('/api/drawing-users/:id', requireAdmin, (req, res) => {
  const username = typeof req.body.username === 'string' ? req.body.username.trim() : '';
  const password = typeof req.body.password === 'string' ? req.body.password : '';
  const validationError = validateDrawingUser(username, password, Boolean(password));
  if (validationError) return res.status(400).json({ error: validationError });
  if (!db.prepare('SELECT id FROM drawing_users WHERE id = ?').get(req.params.id)) return res.status(404).json({ error: 'User not found' });
  try {
    if (password) db.prepare('UPDATE drawing_users SET username = ?, password_hash = ? WHERE id = ?').run(username, hashPassword(password), req.params.id);
    else db.prepare('UPDATE drawing_users SET username = ? WHERE id = ?').run(username, req.params.id);
    revokeDrawingTokens(req.params.id);
    res.json({ success: true, username });
  } catch (error) {
    if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') return res.status(409).json({ error: 'This username already exists' });
    console.error('Update drawing user error:', error);
    res.status(500).json({ error: 'Unable to update user' });
  }
});

app.delete('/api/drawing-users/:id', requireAdmin, (req, res) => {
  const result = db.prepare('DELETE FROM drawing_users WHERE id = ?').run(req.params.id);
  if (!result.changes) return res.status(404).json({ error: 'User not found' });
  revokeDrawingTokens(req.params.id);
  res.json({ success: true });
});

app.patch('/api/drawing-users/:id/status', requireAdmin, (req, res) => {
  const accountStatus = typeof req.body.accountStatus === 'string' ? req.body.accountStatus : '';
  if (!['approved', 'declined', 'blocked'].includes(accountStatus)) return res.status(400).json({ error: 'Invalid account status' });
  const result = db.prepare('UPDATE drawing_users SET account_status = ? WHERE id = ?').run(accountStatus, req.params.id);
  if (!result.changes) return res.status(404).json({ error: 'User not found' });
  revokeDrawingTokens(req.params.id);
  res.json({ success: true, accountStatus });
});

app.post('/api/drawings/login', (req, res) => {
  const username = typeof req.body.username === 'string' ? req.body.username.trim() : '';
  const password = typeof req.body.password === 'string' ? req.body.password : '';
  const user = db.prepare('SELECT id, username, password_hash, account_status FROM drawing_users WHERE username = ?').get(username);
  if (!user || !verifyPassword(password, user.password_hash)) return res.status(401).json({ error: 'Invalid username or password' });
  if (user.account_status === 'pending') return res.status(403).json({ error: 'Your account is waiting for admin approval' });
  if (user.account_status === 'declined') return res.status(403).json({ error: 'Your account request was declined' });
  if (user.account_status === 'blocked') return res.status(403).json({ error: 'Your account has been blocked' });
  const token = crypto.randomBytes(32).toString('hex');
  drawingTokens.set(token, user.id);
  res.json({ token, username: user.username });
});

app.get('/api/drawings/session', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  const userId = token && drawingTokens.get(token);
  const user = userId && db.prepare('SELECT username, account_status FROM drawing_users WHERE id = ?').get(userId);
  if (!user || user.account_status !== 'approved') {
    if (token) drawingTokens.delete(token);
    return res.status(401).json({ error: 'Drawing login required' });
  }
  res.json({ username: user.username });
});

app.post('/api/drawings/logout', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (token) drawingTokens.delete(token);
  res.json({ success: true });
});

app.get('/api/document-types', (req, res) => {
  const query = isAdmin(req)
    ? 'SELECT id, name, requires_login FROM document_types ORDER BY name COLLATE NOCASE'
    : 'SELECT id, name, requires_login FROM document_types WHERE requires_login = 0 ORDER BY name COLLATE NOCASE';
  res.json(db.prepare(query).all());
});

app.post('/api/document-types', requireAdmin, (req, res) => {
  const name = typeof req.body.name === 'string' ? req.body.name.trim() : '';
  const requiresLogin = req.body.requiresLogin ? 1 : 0;
  if (!name) return res.status(400).json({ error: 'Document type name is required' });
  if (name.length > 80) return res.status(400).json({ error: 'Document type name must be 80 characters or fewer' });

  try {
    const result = db.prepare('INSERT INTO document_types (name, requires_login) VALUES (?, ?)').run(name, requiresLogin);
    res.status(201).json({ id: result.lastInsertRowid, name, requiresLogin: Boolean(requiresLogin) });
  } catch (error) {
    if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') return res.status(409).json({ error: 'This document type already exists' });
    console.error('Create document type error:', error);
    res.status(500).json({ error: 'Unable to create document type' });
  }
});

app.patch('/api/document-types/:id', requireAdmin, (req, res) => {
  const name = typeof req.body.name === 'string' ? req.body.name.trim() : '';
  const requiresLogin = req.body.requiresLogin ? 1 : 0;
  if (!name) return res.status(400).json({ error: 'Document type name is required' });
  if (name.length > 80) return res.status(400).json({ error: 'Document type name must be 80 characters or fewer' });

  const documentType = db.prepare('SELECT id, name FROM document_types WHERE id = ?').get(req.params.id);
  if (!documentType) return res.status(404).json({ error: 'Document type not found' });

  try {
    db.transaction(() => {
      db.prepare('UPDATE document_types SET name = ?, requires_login = ? WHERE id = ?').run(name, requiresLogin, documentType.id);
      db.prepare('UPDATE documents SET document_type = ? WHERE document_type = ?').run(name, documentType.name);
    })();
    res.json({ success: true, id: documentType.id, name, requiresLogin: Boolean(requiresLogin) });
  } catch (error) {
    if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') return res.status(409).json({ error: 'This document type already exists' });
    console.error('Update document type error:', error);
    res.status(500).json({ error: 'Unable to update document type' });
  }
});

app.delete('/api/document-types/:id', requireAdmin, (req, res) => {
  const documentType = db.prepare('SELECT id, name FROM document_types WHERE id = ?').get(req.params.id);
  if (!documentType) return res.status(404).json({ error: 'Document type not found' });

  const documentCount = db.prepare('SELECT COUNT(*) AS count FROM documents WHERE document_type = ?').get(documentType.name).count;
  if (documentCount) return res.status(409).json({ error: `This type is used by ${documentCount} document${documentCount === 1 ? '' : 's'}. Reassign or rename those documents before deleting it.` });

  db.prepare('DELETE FROM document_types WHERE id = ?').run(documentType.id);
  res.json({ success: true });
});

app.post('/api/feedback', feedbackUpload.single('attachment'), (req, res) => {
  const message = typeof req.body.message === 'string' ? req.body.message.trim() : '';
  if (!message) {
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    const result = db.prepare(`
      INSERT INTO feedback_submissions (sender_name, sender_contact, message, filename, original_name, file_size, mime_type)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(
      typeof req.body.senderName === 'string' ? req.body.senderName.trim() : '',
      typeof req.body.senderContact === 'string' ? req.body.senderContact.trim() : '',
      message,
      req.file?.filename || '',
      req.file?.originalname || '',
      req.file?.size || null,
      req.file?.mimetype || ''
    );
    res.json({ success: true, id: result.lastInsertRowid });
  } catch (error) {
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    console.error('Feedback error:', error);
    res.status(500).json({ error: 'Unable to save feedback' });
  }
});

function normalizeSearchText(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function editDistance(left, right) {
  const distances = Array.from({ length: left.length + 1 }, () => Array(right.length + 1).fill(0));
  for (let leftIndex = 0; leftIndex <= left.length; leftIndex += 1) distances[leftIndex][0] = leftIndex;
  for (let rightIndex = 0; rightIndex <= right.length; rightIndex += 1) distances[0][rightIndex] = rightIndex;
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      distances[leftIndex][rightIndex] = Math.min(
        distances[leftIndex][rightIndex - 1] + 1,
        distances[leftIndex - 1][rightIndex] + 1,
        distances[leftIndex - 1][rightIndex - 1] + (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1)
      );
      if (leftIndex > 1 && rightIndex > 1 && left[leftIndex - 1] === right[rightIndex - 2] && left[leftIndex - 2] === right[rightIndex - 1]) {
        distances[leftIndex][rightIndex] = Math.min(distances[leftIndex][rightIndex], distances[leftIndex - 2][rightIndex - 2] + 1);
      }
    }
  }
  return distances[left.length][right.length];
}

function searchTermMatches(term, text, textTokens) {
  if (text.includes(term)) return true;
  // Document numbers must retain their exact digit order. Allow partial number matches
  // (for example, 123 matches 1234), but never apply typo correction to numeric terms.
  if (/\d/.test(term)) return false;
  if (term.length < 3) return false;
  const maxDistance = term.length <= 4 ? 1 : term.length <= 8 ? 2 : 3;
  return textTokens.some(token => {
    const distance = editDistance(term, token);
    return distance <= maxDistance && distance / Math.max(term.length, token.length) <= 0.34;
  });
}

// Search each word independently so word order and small typing mistakes do not prevent matches.
app.get('/api/search', (req, res) => {
  const queryTerms = normalizeSearchText(req.query.q).split(' ').filter(Boolean);
  if (!queryTerms.length) return res.json([]);

  const requestedType = typeof req.query.type === 'string' ? req.query.type.trim() : '';
  const typeRecord = requestedType && db.prepare('SELECT name, requires_login FROM document_types WHERE name = ?').get(requestedType);
  if (typeRecord && typeRecord.requires_login && !isDrawingUser(req)) return res.status(401).json({ error: 'Login is required for this document type' });
  const type = typeRecord ? typeRecord.name : '';
  const canAccessProtectedTypes = isDrawingUser(req);
  const stmt = type
    ? db.prepare(`
        SELECT id, filename, original_name, document_number, document_type, keywords, caption, upload_date, file_size
        FROM documents
      WHERE document_type = ?
      `)
    : db.prepare(canAccessProtectedTypes ? `
        SELECT id, filename, original_name, document_number, document_type, keywords, caption, upload_date, file_size
        FROM documents
      ` : `
        SELECT d.id, d.filename, d.original_name, d.document_number, d.document_type, d.keywords, d.caption, d.upload_date, d.file_size
        FROM documents d
        LEFT JOIN document_types dt ON d.document_type = dt.name
        WHERE COALESCE(dt.requires_login, 0) = 0
      `);
  const documents = type ? stmt.all(type) : stmt.all();

  const results = documents
    .map(document => {
      const text = normalizeSearchText(`${document.document_number} ${document.caption} ${document.keywords} ${document.original_name}`);
      const textTokens = text.split(' ').filter(Boolean);
      const matches = queryTerms.every(term => searchTermMatches(term, text, textTokens));
      const exactMatches = queryTerms.filter(term => text.includes(term)).length;
      return matches ? { document, exactMatches } : null;
    })
    .filter(Boolean)
    .sort((left, right) => right.exactMatches - left.exactMatches || String(right.document.upload_date).localeCompare(String(left.document.upload_date)))
    .slice(0, 50)
    .map(result => result.document);

  res.json(results);
});

// Get all documents (for admin/upload page)
app.get('/api/documents', requireAdmin, (req, res) => {
  const stmt = db.prepare(`
    SELECT id, filename, original_name, document_number, document_type, keywords, caption, upload_date, file_size
    FROM documents
    ORDER BY upload_date DESC
  `);
  const results = stmt.all();
  res.json(results);
});

app.get('/api/feedback', requireAdmin, (req, res) => {
  const submissions = db.prepare(`
    SELECT id, sender_name, sender_contact, message, original_name, file_size, mime_type, submitted_at,
      CASE WHEN filename = '' THEN 0 ELSE 1 END AS has_file
    FROM feedback_submissions
    ORDER BY submitted_at DESC
  `).all();
  res.json(submissions);
});

app.delete('/api/feedback/:id', requireAdmin, (req, res) => {
  const submission = db.prepare('SELECT filename FROM feedback_submissions WHERE id = ?').get(req.params.id);
  if (!submission) return res.status(404).json({ error: 'Feedback not found' });

  if (submission.filename) {
    const filePath = path.join(feedbackUploadDirectory, submission.filename);
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  }
  db.prepare('DELETE FROM feedback_submissions WHERE id = ?').run(req.params.id);
  res.json({ success: true });
});

// Upload document
app.post('/api/upload', requireAdmin, upload.single('pdf'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const { caption, documentNumber } = req.body;
    const documentType = typeof req.body.documentType === 'string' ? req.body.documentType.trim() : '';
    const keywords = typeof req.body.keywords === 'string' ? req.body.keywords.trim() : '';
    if (!documentNumber || documentNumber.trim() === '') {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Document number is required' });
    }
    if (!documentType || !db.prepare('SELECT 1 FROM document_types WHERE name = ?').get(documentType)) {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Please select a valid document type' });
    }
    if (!caption || caption.trim() === '') {
      // Delete the uploaded file if no caption
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Caption is required' });
    }
    
    const stmt = db.prepare(`
      INSERT INTO documents (filename, original_name, document_number, document_type, keywords, caption, file_size, mime_type)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    
    const result = stmt.run(
      req.file.filename,
      req.file.originalname,
      documentNumber.trim(),
      documentType,
      keywords,
      caption.trim(),
      req.file.size,
      req.file.mimetype
    );
    
    res.json({
      success: true,
      id: result.lastInsertRowid,
      message: 'Document uploaded successfully'
    });
  } catch (error) {
    console.error('Upload error:', error);
    // Clean up file if error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({ error: error.message });
  }
});

// View document inline in the browser
app.get('/api/view/:id', (req, res) => {
  const stmt = db.prepare('SELECT filename, original_name FROM documents WHERE id = ?');
  const doc = stmt.get(req.params.id);

  if (!doc) return res.status(404).json({ error: 'Document not found' });

  const filePath = path.join(uploadDirectory, doc.filename);
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'File not found on server' });

  res.type('application/pdf');
  res.set('Content-Disposition', 'inline');
  res.sendFile(filePath);
});

// Download document
app.get('/api/download/:id', (req, res) => {
  const { id } = req.params;
  
  const stmt = db.prepare('SELECT filename, original_name FROM documents WHERE id = ?');
  const doc = stmt.get(id);
  
  if (!doc) {
    return res.status(404).json({ error: 'Document not found' });
  }
  
  const filePath = path.join(uploadDirectory, doc.filename);
  
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'File not found on server' });
  }
  
  res.download(filePath, doc.original_name);
});

app.get('/api/feedback/:id/download', requireAdmin, (req, res) => {
  const submission = db.prepare('SELECT filename, original_name FROM feedback_submissions WHERE id = ?').get(req.params.id);
  if (!submission || !submission.filename) return res.status(404).json({ error: 'Feedback file not found' });

  const filePath = path.join(feedbackUploadDirectory, submission.filename);
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Feedback file not found on server' });
  res.download(filePath, submission.original_name);
});

app.patch('/api/documents/:id', requireAdmin, (req, res) => {
  const documentType = typeof req.body.documentType === 'string' ? req.body.documentType.trim() : '';
  const documentNumber = typeof req.body.documentNumber === 'string' ? req.body.documentNumber.trim() : '';
  const caption = typeof req.body.caption === 'string' ? req.body.caption.trim() : '';
  const keywords = typeof req.body.keywords === 'string' ? req.body.keywords.trim() : '';
  if (!documentType || !documentNumber || !caption) return res.status(400).json({ error: 'Document type, number and caption are required' });
  if (!db.prepare('SELECT 1 FROM document_types WHERE name = ?').get(documentType)) return res.status(400).json({ error: 'Please select a valid document type' });

  const result = db.prepare(`
    UPDATE documents
    SET document_type = ?, document_number = ?, caption = ?, keywords = ?
    WHERE id = ?
  `).run(documentType, documentNumber, caption, keywords, req.params.id);
  if (!result.changes) return res.status(404).json({ error: 'Document not found' });
  res.json({ success: true });
});

// Delete document (optional admin feature)
app.delete('/api/documents/:id', requireAdmin, (req, res) => {
  const { id } = req.params;
  
  const stmt = db.prepare('SELECT filename FROM documents WHERE id = ?');
  const doc = stmt.get(id);
  
  if (!doc) {
    return res.status(404).json({ error: 'Document not found' });
  }
  
  // Delete file
  const filePath = path.join(uploadDirectory, doc.filename);
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
  
  // Delete from database
  db.prepare('DELETE FROM documents WHERE id = ?').run(id);
  
  res.json({ success: true, message: 'Document deleted' });
});

// Serve the main app for all other routes (SPA support)
// Check for admin authentication if the route contains 'admin' or 'upload'
app.get('*', (req, res) => {
  const pathName = req.path || '';
  const isAdminRoute = pathName.includes('admin') || pathName.includes('upload') || pathName === '';
  
  if (isAdminRoute && !ADMIN_PASSWORD) {
    return res.status(503).json({ error: 'Admin access is not configured on the server' });
  }
  
  if (isAdminRoute) {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token || !adminTokens.has(token)) {
      // Return 401 with JSON error so frontend can show login form
      return res.status(401).json({ error: 'Admin login required' });
    }
  }
  
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`SMI MS TC Web App running on http://localhost:${PORT}`);
  console.log(`Upload directory: ${path.join(__dirname, 'uploads')}`);
});
