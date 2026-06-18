let debugInterval = null;
let debugPaused   = false;

document.addEventListener('DOMContentLoaded', () => {
    loadUsers();
    loadLogs();
    setInterval(loadLogs, 5000);
    startDebugPoll();
});


// ── Registration ──────────────────────────────────────────────────────────────

async function registerUser() {
    const name = document.getElementById('userName').value.trim();
    if (!name) { showToast('Please enter a name', 'error'); return; }

    updateStatus('Capturing face...', 'warning');

    try {
        const capData = await post('/capture_frame');
        if (!capData.success) {
            showToast(capData.message, 'error');
            updateStatus('Ready');
            return;
        }

        updateStatus('Registering...', 'warning');
        const regData = await post('/register', { name, face_data: capData.face_data });

        if (regData.success) {
            showToast(`${name} registered!`, 'success');
            document.getElementById('userName').value = '';
            loadUsers();
        } else {
            showToast(regData.message, 'error');
        }
    } catch (e) {
        showToast('Registration failed: ' + e.message, 'error');
    }
    updateStatus('Ready');
}


// ── Recognition ───────────────────────────────────────────────────────────────

async function recognizeUser() {
    updateStatus('Scanning...', 'warning');

    try {
        const data = await post('/recognize');
        const rd   = document.getElementById('resultDisplay');

        if (data.success) {
            rd.className = 'result-display success show';
            rd.innerHTML = `
                <div class="result-name">&#10003; ${data.name}</div>
                <div class="result-confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                <div class="result-confidence">${data.timestamp}</div>`;
            showToast(`Attendance marked for ${data.name}!`, 'success');
            updateStatus('Recognized', 'success');
            loadLogs();
        } else {
            rd.className = 'result-display error show';
            rd.innerHTML = `
                <div class="result-name">&#10007; ${data.message}</div>
                ${data.confidence
                    ? `<div class="result-confidence">Best match: ${(data.confidence * 100).toFixed(1)}%</div>`
                    : ''}`;
            showToast(data.message, 'error');
            updateStatus('Ready');
        }

        setTimeout(() => {
            rd.classList.remove('show');
            updateStatus('Ready');
        }, 5000);

    } catch (e) {
        showToast('Scan failed: ' + e.message, 'error');
        updateStatus('Ready');
    }
}


// ── Data loaders ──────────────────────────────────────────────────────────────

async function loadUsers() {
    try {
        const data = await (await fetch('/get_users')).json();
        document.getElementById('userCount').textContent = data.count;
        const ul = document.getElementById('usersList');
        ul.innerHTML = data.users.length === 0
            ? '<p class="text-muted">No users registered yet</p>'
            : data.users.map(u => `
                <div class="user-item">
                    <div class="user-avatar">${u.charAt(0).toUpperCase()}</div>
                    <div class="user-name">${u}</div>
                </div>`).join('');
    } catch (_) {}
}

async function loadLogs() {
    try {
        const data = await (await fetch('/get_logs')).json();
        const al = document.getElementById('activityLog');
        if (!data.logs.length) {
            al.innerHTML = '<p class="text-muted">No activity yet</p>';
            return;
        }
        al.innerHTML = data.logs.slice(-10).reverse().map(log => `
            <div class="activity-item success">
                <div class="activity-header">
                    <div class="activity-name">&#128203; ${log.name}</div>
                    <div class="activity-time">${new Date(log.timestamp).toLocaleTimeString()}</div>
                </div>
                <div class="activity-details">
                    Attendance marked &middot; Confidence: ${(log.confidence * 100).toFixed(1)}%
                </div>
            </div>`).join('');
    } catch (_) {}
}


// ── Debug panel ───────────────────────────────────────────────────────────────

function startDebugPoll() {
    if (debugInterval) return;
    debugInterval = setInterval(fetchDebugScores, 1500);
}

function toggleDebugPoll() {
    debugPaused = !debugPaused;
    const icon = document.getElementById('debugToggleIcon');
    icon.innerHTML = debugPaused
        ? '<polygon points="5 3 19 12 5 21 5 3"></polygon>'
        : '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    if (!debugPaused) fetchDebugScores();
}

async function fetchDebugScores() {
    if (debugPaused) return;
    try {
        const data = await (await fetch('/debug_scores')).json();
        renderDebugPanel(data);
    } catch (_) {}
}

function renderDebugPanel(data) {
    const panel = document.getElementById('debugPanel');
    if (!panel || !data || !data.label) {
        return;   // no scan yet — keep placeholder text
    }

    const isReal = data.is_real;
    const conf   = Math.round(data.confidence * 100);
    const score  = data.scores && data.scores.antispoof_score !== undefined
        ? data.scores.antispoof_score.toFixed(3)
        : '—';

    panel.innerHTML = `
        <div class="debug-verdict ${isReal ? 'real' : 'spoof'}">
            ${isReal ? '&#10003; REAL' : '&#10007; SPOOF'} &nbsp;&middot;&nbsp; ${conf}%
        </div>
        <div class="debug-section-label">FasNet score</div>
        <div class="debug-row">
            <span class="debug-label">antispoof_score</span>
            <div class="debug-bar-track">
                <div class="debug-bar-fill" style="width:${Math.round(parseFloat(score)*100)}%;background:${parseFloat(score) > 0.5 ? '#10b981' : '#ef4444'}"></div>
            </div>
            <span class="debug-value">${score}</span>
        </div>
        <div class="debug-threshold-note">score &gt; 0.5 → Real &nbsp;|&nbsp; score &lt; 0.5 → Spoof</div>`;
}


// ── Utilities ─────────────────────────────────────────────────────────────────

async function post(url, body) {
    const opts = { method: 'POST' };
    if (body) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body    = JSON.stringify(body);
    }
    return (await fetch(url, opts)).json();
}

function updateStatus(message, type = 'success') {
    const badge = document.getElementById('statusBadge');
    badge.textContent = message;
    badge.className   = 'status-badge';
    if (type === 'warning') badge.classList.add('warning');
    else if (type === 'error') badge.classList.add('error');
}

function showToast(message, type = 'success') {
    document.getElementById('toastMessage').textContent = message;
    document.getElementById('toast').classList.add('show');
    setTimeout(() => document.getElementById('toast').classList.remove('show'), 3000);
}

async function quitApp() {
    if (!confirm('Stop the server and quit?')) return;
    try {
        await post('/shutdown');
    } catch (_) {}
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;color:#666;font-size:1.2rem;">Server stopped. You can close this tab.</div>';
}
