const fs = require('fs');
const path = require('path');

function parseFile(filePath) {
    const fileBuffer = fs.readFileSync(filePath);
    const fileContent = fileBuffer.toString();
    const lines = fileContent.split('\n');

    const data = [];
    let currentExample = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line.startsWith('#') || line === '') {
            continue;
        }

        const values = line.split(',');

        if (values.length === 0) {
            continue;
        }

        if (currentExample.length > 0) {
            data.push(currentExample);
            currentExample = [];
        }

        for (let j = 0; j < values.length; j++) {
            currentExample.push(parseFloat(values[j]));
        }
    }

    if (currentExample.length > 0) {
        data.push(currentExample);
    }

    return data;
}

function parseDirectory(directoryPath) {
    const files = fs.readdirSync(directoryPath);

    const data = [];
    for (const file of files) {
        const filePath = path.join(directoryPath, file);
        const fileStat = fs.statSync(filePath);

        if (fileStat.isDirectory()) {
            const subdirectoryData = parseDirectory(filePath);
            data.push(...subdirectoryData);
        } else if (file.endsWith('.csv')) {
            const fileData = parseFile(filePath);
            data.push(...fileData);
        }
    }

    return data;
}

module.exports = { parseFile, parseDirectory };