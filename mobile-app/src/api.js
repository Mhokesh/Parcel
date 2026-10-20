import Constants from 'expo-constants';
export function discoverServer() {
  if (process.env.EXPO_PUBLIC_API_URL) return process.env.EXPO_PUBLIC_API_URL.replace(/\/+$/, '');
  const host = Constants.expoConfig?.hostUri || Constants.expoGoConfig?.debuggerHost || '';
  if (!host || /exp\.direct|expo\.dev/.test(host)) return '';
  const hostname = host.replace(/^https?:\/\//, '').split(':')[0];
  return hostname ? `http://${hostname}:8000` : '';
}
export function validateServer(value) {
  const url = new URL(value.trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.pathname !== '/') throw new Error('Use a server address such as http://192.168.1.10:8000');
  return url.origin;
}
export async function request(server, key, path, body) {
  if (!server) throw new Error('Add your computer’s server address in Settings.');
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 45000);
  try {
    const response = await fetch(`${server}${path}`, { method: body ? 'POST' : 'GET', headers: { 'Content-Type': 'application/json', 'X-API-Key': key || '' }, body: body ? JSON.stringify(body) : undefined, signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Server returned ${response.status}`);
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Connection timed out. Keep the server running and check your Wi-Fi.');
    if (error.message === 'Network request failed') throw new Error('Could not reach the server. Check the address, Wi-Fi and local-network permission in iPhone Settings.');
    throw error;
  } finally { clearTimeout(timer); }
}
