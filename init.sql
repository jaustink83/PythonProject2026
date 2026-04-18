-- init.sql
-- Runs when the MySQL container first starts.

CREATE DATABASE IF NOT EXISTS gradepredictor;

USE gradepredictor;

-- ── Users ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            INT          NOT NULL AUTO_INCREMENT,
    username      VARCHAR(80)  NOT NULL,
    email         VARCHAR(120) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_users_username (username),
    UNIQUE KEY uq_users_email    (email)
);

-- ── Predictions ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              INT         NOT NULL AUTO_INCREMENT,
    user_id         INT         NOT NULL,
    inputs_json     TEXT        NOT NULL,   -- JSON blob of the user's inputs
    predicted_score TINYINT     NOT NULL,   -- 0–20
    grade_letter    VARCHAR(2)  NOT NULL,   -- A / B / C / D / F
    percentage      TINYINT     NOT NULL,   -- 0–100
    created_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_predictions_user_id (user_id),
    CONSTRAINT fk_predictions_user
        FOREIGN KEY (user_id) REFERENCES users (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
