// Keep Expo's event log outside OneDrive: a recalled placeholder can crash its logger.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const logDir = path.join(os.tmpdir(), 'parcel-expo-logs');
fs.mkdirSync(logDir, { recursive: true });
const logFile = path.join(logDir, `start-${Date.now()}-${process.pid}.log`);
const expoCli = path.join(path.dirname(require.resolve('expo/package.json')), 'bin', 'cli');
const result = spawnSync(process.execPath, [expoCli, 'start', '--go', ...process.argv.slice(2)], {
  cwd: path.resolve(__dirname, '..'),
  stdio: 'inherit',
  env: { ...process.env, LOG_EVENTS: logFile },
});
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
