const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const Database = require('better-sqlite3');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use('/uploads', express.static('uploads'));

// Database setup
const db = new Database('smi_tc.db');

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

// Multer storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
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

// Search documents
app.get('/api/search', (req, res) => {
  const { q } = req.query;
  
  if (!q || q.trim() === '') {
    return res.json([]);
  }
  
  const searchTerm = `%${q.trim()}%`;
  const stmt = db.prepare(`
    SELECT id, filename, original_name, caption, upload_date, file_size
    FROM documents
    WHERE caption LIKE ? OR original_name LIKE ?
    ORDER BY upload_date DESC
    LIMIT 50
  `);
  
  const results = stmt.all(searchTerm, searchTerm);
  res.json(results);
});

// Get all documents (for admin/upload page)
app.get('/api/documents', (req, res) => {
  const stmt = db.prepare(`
    SELECT id, filename, original_name, caption, upload_date, file_size
    FROM documents
    ORDER BY upload_date DESC
  `);
  const results = stmt.all();
  res.json(results);
});

// Upload document
app.post('/api/upload', upload.single('pdf'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const { caption } = req.body;
    if (!caption || caption.trim() === '') {
      // Delete the uploaded file if no caption
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Caption is required' });
    }
    
    const stmt = db.prepare(`
      INSERT INTO documents (filename, original_name, caption, file_size, mime_type)
      VALUES (?, ?, ?, ?, ?)
    `);
    
    const result = stmt.run(
      req.file.filename,
      req.file.originalname,
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
  
  const filePath = path.join(__dirname, 'uploads', doc.filename);
  
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'File not found on server' });
  }
  
  res.download(filePath, doc.original_name);
});

// Delete document (optional admin feature)
app.delete('/api/documents/:id', (req, res) => {
  const { id } = req.params;
  
  const stmt = db.prepare('SELECT filename FROM documents WHERE id = ?');
  const doc = stmt.get(id);
  
  if (!doc) {
    return res.status(404).json({ error: 'Document not found' });
  }
  
  // Delete file
  const filePath = path.join(__dirname, 'uploads', doc.filename);
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