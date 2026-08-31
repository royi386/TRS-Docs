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
const databasePath = process.env.DB_PATH || path.join(__dirname, 'smi_tc.db');
fs.mkdirSync(uploadDirectory, { recursive: true });

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

// Search documents
app.get('/api/search', (req, res) => {
  const { q } = req.query;
  
  if (!q || q.trim() === '') {
    return res.json([]);
  }
  
  const searchTerm = `%${q.trim()}%`;
  const stmt = db.prepare(`
    SELECT id, filename, original_name, document_number, caption, upload_date, file_size
    FROM documents
    WHERE document_number LIKE ? OR caption LIKE ? OR original_name LIKE ?
    ORDER BY upload_date DESC
    LIMIT 50
  `);
  
  const results = stmt.all(searchTerm, searchTerm, searchTerm);
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

// Upload document
app.post('/api/upload', requireAdmin, upload.single('pdf'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const { caption, documentNumber } = req.body;
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
      INSERT INTO documents (filename, original_name, document_number, caption, file_size, mime_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    
    const result = stmt.run(
      req.file.filename,
      req.file.originalname,
      documentNumber.trim(),
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
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`SMI MS TC Web App running on http://localhost:${PORT}`);
  console.log(`Upload directory: ${path.join(__dirname, 'uploads')}`);
});