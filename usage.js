'use strict';

/**
 * Usage logging for the admin dashboard: page hits and AI questions.
 *
 * Two tables keep the dashboard queries dead simple:
 *   usage_page_hits  — one row per page view (name, day, hour)
 *   usage_questions  — one row per AI question (day, hour, question, cached)
 *
 * Day/hour are denormalised (local server time, hour as 0-23) so the graph
 * queries are plain GROUP BYs with no date arithmetic. Raw question text is
 * capped and pruned: the aggregates keep every hit forever (a few bytes per
 * row), while the readable log keeps the last few thousand questions.
 */

const USAGE_QUESTION_LOG_MAX = 5000;

function createUsageLog(db, logger = console) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS usage_page_hits (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      day TEXT NOT NULL,
      hour INTEGER NOT NULL,
      at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_usage_hits_day ON usage_page_hits(day);
    CREATE INDEX IF NOT EXISTS idx_usage_hits_name ON usage_page_hits(name);

    CREATE TABLE IF NOT EXISTS usage_questions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      day TEXT NOT NULL,
      hour INTEGER NOT NULL,
      question TEXT NOT NULL,
      model TEXT NOT NULL DEFAULT '',
      cached INTEGER NOT NULL DEFAULT 0,
      duration_ms INTEGER,
      at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_usage_questions_day ON usage_questions(day);
    CREATE INDEX IF NOT EXISTS idx_usage_questions_at ON usage_questions(at);
  `);

  const statements = {
    insertHit: db.prepare('INSERT INTO usage_page_hits (name, day, hour) VALUES (?, ?, ?)'),
    insertQuestion: db.prepare('INSERT INTO usage_questions (day, hour, question, model, cached, duration_ms) VALUES (?, ?, ?, ?, ?, ?)'),
    pruneQuestions: db.prepare('DELETE FROM usage_questions WHERE id NOT IN (SELECT id FROM usage_questions ORDER BY id DESC LIMIT ?)'),
    counts: db.prepare(`
      SELECT
        (SELECT COUNT(*) FROM usage_page_hits) AS total_hits,
        (SELECT COUNT(*) FROM usage_page_hits WHERE day = @today) AS hits_today,
        (SELECT COUNT(*) FROM usage_questions) AS total_questions,
        (SELECT COUNT(*) FROM usage_questions WHERE day = @today) AS questions_today,
        (SELECT COUNT(*) FROM usage_questions WHERE day = @today AND cached = 0) AS questions_today_generated,
        (SELECT COUNT(*) FROM usage_questions WHERE day = @today AND cached = 1) AS questions_today_cached,
        (SELECT AVG(duration_ms) FROM usage_questions WHERE day = @today AND duration_ms IS NOT NULL AND cached = 0) AS avg_generated_ms
    `),
    perDay: db.prepare(`
      SELECT day,
             SUM(CASE WHEN kind = 'hit' THEN 1 ELSE 0 END) AS hits,
             SUM(CASE WHEN kind = 'question' THEN 1 ELSE 0 END) AS questions
      FROM (
        SELECT day, 'hit' AS kind FROM usage_page_hits
        UNION ALL
        SELECT day, 'question' AS kind FROM usage_questions
      )
      WHERE day >= @since
      GROUP BY day
      ORDER BY day
    `),
    questionsPerDay: db.prepare(`
      SELECT day, COUNT(*) AS questions,
             SUM(cached) AS cached
      FROM usage_questions
      WHERE day >= @since
      GROUP BY day
      ORDER BY day
    `),
    hitsPerHour: db.prepare(`
      SELECT hour, COUNT(*) AS hits
      FROM usage_page_hits
      WHERE day = @today
      GROUP BY hour
      ORDER BY hour
    `),
    questionsPerHour: db.prepare(`
      SELECT hour, COUNT(*) AS questions,
             SUM(cached) AS cached
      FROM usage_questions
      WHERE day = @today
      GROUP BY hour
      ORDER BY hour
    `),
    recentQuestions: db.prepare('SELECT at, question, model, cached, duration_ms AS durationMs FROM usage_questions ORDER BY id DESC LIMIT ?'),
    topQuestions: db.prepare(`
      SELECT LOWER(TRIM(question)) AS question, COUNT(*) AS count
      FROM usage_questions
      WHERE day >= @since
      GROUP BY LOWER(TRIM(question))
      ORDER BY count DESC
      LIMIT ?
    `),
    topPages: db.prepare(`
      SELECT name, COUNT(*) AS count
      FROM usage_page_hits
      WHERE day >= @since
      GROUP BY name
      ORDER BY count DESC
      LIMIT ?
    `),
    activeDays: db.prepare("SELECT COUNT(DISTINCT day) AS days FROM usage_page_hits WHERE day >= @since")
  };

  function dayString(now = new Date()) {
    const pad = number => String(number).padStart(2, '0');
    return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  }

  function recordPageHit(name) {
    try {
      const now = new Date();
      statements.insertHit.run(String(name || 'unknown').slice(0, 80), dayString(now), now.getHours());
    } catch (error) {
      logger.warn(`[usage] page hit not recorded: ${error.message}`);
    }
  }

  function recordQuestion({ question, model = '', cached = false, durationMs = null } = {}) {
    try {
      const now = new Date();
      statements.insertQuestion.run(
        dayString(now),
        now.getHours(),
        String(question || '').slice(0, 1000),
        String(model || '').slice(0, 120),
        cached ? 1 : 0,
        Number.isFinite(durationMs) ? Math.round(durationMs) : null
      );
      // Cheap prune; the delete touches nothing when under the cap.
      statements.pruneQuestions.run(USAGE_QUESTION_LOG_MAX);
    } catch (error) {
      logger.warn(`[usage] question not recorded: ${error.message}`);
    }
  }

  function dayOffset(days) {
    const date = new Date();
    date.setDate(date.getDate() - days);
    return dayString(date);
  }

  /**
   * The dashboard payload: totals, per-day series for the big graph,
   * per-hour series for today, the readable question log and the top lists.
   * rangeDays selects how far the daily graph reaches.
   */
  function summary({ rangeDays = 14, logLimit = 50 } = {}) {
    const today = dayString();
    const since = dayOffset(Math.max(1, Math.min(rangeDays, 90)) - 1);
    const counts = statements.counts.get({ today });

    const fillDays = (rows, key) => {
      const byDay = new Map(rows.map(row => [row.day, row]));
      const series = [];
      for (let offset = Math.max(1, Math.min(rangeDays, 90)) - 1; offset >= 0; offset -= 1) {
        const day = dayOffset(offset);
        series.push({ day, [key]: Number(byDay.get(day)?.[key] || 0) });
      }
      return series;
    };

    // Hits and questions arrive as separate rows per hour; merge them by key
    // instead of letting one overwrite the other.
    const fillHours = rows => {
      const byHour = new Map();
      for (const row of rows) {
        const entry = byHour.get(row.hour) || { hour: row.hour, hits: 0, questions: 0 };
        if (row.hits !== undefined) entry.hits = Number(row.hits);
        if (row.questions !== undefined) entry.questions = Number(row.questions);
        byHour.set(row.hour, entry);
      }
      return Array.from({ length: 24 }, (_, hour) => byHour.get(hour) || { hour, hits: 0, questions: 0 });
    };

    const daily = fillDays(statements.perDay.all({ since }), 'hits');
    const questionDaily = statements.questionsPerDay.all({ since });
    const perHourRows = [...statements.hitsPerHour.all({ today }), ...statements.questionsPerHour.all({ today })];
    const perHour = fillHours(perHourRows);

    return {
      totals: {
        pageHits: counts.total_hits || 0,
        pageHitsToday: counts.hits_today || 0,
        questions: counts.total_questions || 0,
        questionsToday: counts.questions_today || 0,
        questionsTodayGenerated: counts.questions_today_generated || 0,
        questionsTodayCached: counts.questions_today_cached || 0,
        avgGeneratedMs: counts.avg_generated_ms ? Math.round(counts.avg_generated_ms) : null,
        activeDays: statements.activeDays.get({ since }).days || 0
      },
      rangeDays: Math.max(1, Math.min(rangeDays, 90)),
      daily: daily.map((entry, index) => ({
        day: entry.day,
        hits: entry.hits,
        questions: Number(questionDaily.find(row => row.day === entry.day)?.questions || 0)
      })),
      perHour,
      recentQuestions: statements.recentQuestions.all(Math.max(1, Math.min(logLimit, 200))),
      topQuestions: statements.topQuestions.all({ since }, 10),
      topPages: statements.topPages.all({ since }, 10)
    };
  }

  return { recordPageHit, recordQuestion, summary };
}

module.exports = { createUsageLog };
