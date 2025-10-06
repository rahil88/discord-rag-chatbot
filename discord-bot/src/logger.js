const winston = require('winston');
const path = require('path');

// Define log levels
const logLevels = {
    error: 0,
    warn: 1,
    info: 2,
    http: 3,
    debug: 4,
};

// Define log colors
const logColors = {
    error: 'red',
    warn: 'yellow',
    info: 'green',
    http: 'magenta',
    debug: 'white',
};

// Add colors to winston
winston.addColors(logColors);

// Define log format
const logFormat = winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss:ms' }),
    winston.format.colorize({ all: true }),
    winston.format.printf(
        (info) => `${info.timestamp} ${info.level}: ${info.message}`
    )
);

// Define transports
const transports = [
    // Console transport
    new winston.transports.Console({
        format: logFormat,
    }),
    // File transport for errors
    new winston.transports.File({
        filename: path.join(__dirname, '../logs/error.log'),
        level: 'error',
        format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.errors({ stack: true }),
            winston.format.json()
        ),
    }),
    // File transport for all logs
    new winston.transports.File({
        filename: path.join(__dirname, '../logs/combined.log'),
        format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.errors({ stack: true }),
            winston.format.json()
        ),
    }),
];

// Create the logger
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    levels: logLevels,
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports,
    // Handle exceptions and rejections
    exceptionHandlers: [
        new winston.transports.File({ 
            filename: path.join(__dirname, '../logs/exceptions.log') 
        })
    ],
    rejectionHandlers: [
        new winston.transports.File({ 
            filename: path.join(__dirname, '../logs/rejections.log') 
        })
    ],
});

// Create logs directory if it doesn't exist
const fs = require('fs');
const logsDir = path.join(__dirname, '../logs');
if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
}

module.exports = logger;
