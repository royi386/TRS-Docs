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

function isAdmin(req) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  return Boolean(token && adminTokens.has(token));
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

  const type = ['SMI', 'MS', 'TC', 'Drawings'].includes(req.query.type) ? req.query.type : '';
  const stmt = type
    ? db.prepare(`
        SELECT id, filename, original_name, document_number, caption, upload_date, file_size
        FROM documents
        WHERE document_type = ? OR (document_type = '' AND document_number LIKE ?)
      `)
    : db.prepare(`
        SELECT id, filename, original_name, document_number, caption, upload_date, file_size
        FROM documents
      `);
  const documents = type ? stmt.all(type, `${type}%`) : stmt.all();

  const results = documents
    .map(document => {
      const text = normalizeSearchText(`${document.document_number} ${document.caption} ${document.original_name}`);
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
    SELECT id, filename, original_name, document_number, caption, upload_date, file_size
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
    const documentType = ['SMI', 'MS', 'TC', 'Drawings'].includes(req.body.documentType) ? req.body.documentType : '';
    if (!documentNumber || documentNumber.trim() === '') {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Document number is required' });
    }
    if (!caption || caption.trim() === '') {
      // Delete the uploaded file if no caption
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Caption is required' });
    }
    
    const stmt = db.prepare(`
      INSERT INTO documents (filename, original_name, document_number, document_type, caption, file_size, mime_type)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    
    const result = stmt.run(
      req.file.filename,
      req.file.originalname,
      documentNumber.trim(),
      documentType,
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