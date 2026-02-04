document.addEventListener('DOMContentLoaded', () => {
    loadUsers();
    loadLogs();
    setInterval(loadLogs, 5000);
});

async function registerUser() {
    const name = document.getElementById('userName').value.trim();
    
    if (!name) {
        showToast('Please enter a name', 'error');
        return;
    }
    
    updateStatus('Capturing face...', 'warning');
    
    try {
        const captureResponse = await fetch('/capture_frame', {method: 'POST'});
        const captureData = await captureResponse.json();
        
        if (!captureData.success) {
            showToast(captureData.message, 'error');
            updateStatus('Ready');
            return;
        }
        
        updateStatus('Registering user...', 'warning');
        
        const registerResponse = await fetch('/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                name: name,
                face_data: captureData.face_data
            })
        });
        
        const registerData = await registerResponse.json();
        
        if (registerData.success) {
            showToast(`${name} registered successfully!`, 'success');
            document.getElementById('userName').value = '';
            loadUsers();
            updateStatus('Ready');
        } else {
            showToast(registerData.message, 'error');
            updateStatus('Ready');
        }
        
    } catch (error) {
        showToast('Registration failed: ' + error.message, 'error');
        updateStatus('Ready');
    }
}

async function recognizeUser() {
    updateStatus('Scanning face...', 'warning');
    
    try {
        const response = await fetch('/recognize', {method: 'POST'});
        const data = await response.json();
        
        const resultDisplay = document.getElementById('resultDisplay');
        
        if (data.success) {
            resultDisplay.className = 'result-display success show';
            resultDisplay.innerHTML = `
                <div class="result-name">✓ ${data.name}</div>
                <div class="result-confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                <div class="result-confidence">${data.timestamp}</div>
            `;
            
            showToast(`Attendance marked for ${data.name}!`, 'success');
            updateStatus('Recognized', 'success');
            loadLogs();
            
            setTimeout(() => {
                resultDisplay.classList.remove('show');
                updateStatus('Ready');
            }, 5000);
            
        } else {
            resultDisplay.className = 'result-display error show';
            resultDisplay.innerHTML = `
                <div class="result-name">✗ ${data.message}</div>
                ${data.confidence ? `<div class="result-confidence">Best match: ${(data.confidence * 100).toFixed(1)}%</div>` : ''}
            `;
            
            showToast(data.message, 'error');
            updateStatus('Ready');
            
            setTimeout(() => {
                resultDisplay.classList.remove('show');
            }, 5000);
        }
        
    } catch (error) {
        showToast('Recognition failed: ' + error.message, 'error');
        updateStatus('Ready');
    }
}

async function loadUsers() {
    try {
        const response = await fetch('/get_users');
        const data = await response.json();
        
        document.getElementById('userCount').textContent = data.count;
        
        const usersList = document.getElementById('usersList');
        
        if (data.users.length === 0) {
            usersList.innerHTML = '<p class="text-muted">No users registered yet</p>';
            return;
        }
        
        usersList.innerHTML = data.users.map(user => `
            <div class="user-item">
                <div class="user-avatar">${user.charAt(0).toUpperCase()}</div>
                <div class="user-name">${user}</div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load users:', error);
    }
}

async function loadLogs() {
    try {
        const response = await fetch('/get_logs');
        const data = await response.json();
        
        const activityLog = document.getElementById('activityLog');
        
        if (data.logs.length === 0) {
            activityLog.innerHTML = '<p class="text-muted">No activity yet</p>';
            return;
        }
        
        const recentLogs = data.logs.slice(-10).reverse();
        
        activityLog.innerHTML = recentLogs.map(log => {
            const date = new Date(log.timestamp);
            const timeStr = date.toLocaleTimeString();
            
            return `
                <div class="activity-item success">
                    <div class="activity-header">
                        <div class="activity-name">📋 ${log.name}</div>
                        <div class="activity-time">${timeStr}</div>
                    </div>
                    <div class="activity-details">
                        Attendance marked • Confidence: ${(log.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        console.error('Failed to load logs:', error);
    }
}

function updateStatus(message, type = 'success') {
    const badge = document.getElementById('statusBadge');
    badge.textContent = message;
    badge.className = 'status-badge';
    
    if (type === 'warning') {
        badge.classList.add('warning');
    } else if (type === 'error') {
        badge.classList.add('error');
    }
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    
    toastMessage.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}