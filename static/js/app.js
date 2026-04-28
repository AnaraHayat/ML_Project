// ============================================
// MINDGUARD PRO AI - FRONTEND LOGIC
// ============================================

// --- 1. GLOBAL STATE ---
let currentResult = null;
let analysisHistory = [];

// --- 2. DOM ELEMENTS ---
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const userInput = document.getElementById('userInput');
const age = document.getElementById('age');
const gender = document.getElementById('gender');
const height = document.getElementById('height');
const weight = document.getElementById('weight');
const sleepSlider = document.getElementById('sleep');
const sleepValue = document.getElementById('sleepValue');
const stressSlider = document.getElementById('stress');
const stressValue = document.getElementById('stressValue');
const severitySlider = document.getElementById('severity');
const severityValue = document.getElementById('severityValue');
const mood_scoreSlider = document.getElementById('mood_score');
const mood_scoreValue = document.getElementById('mood_scoreValue');
const lifestyle = document.getElementById('lifestyle');
const emotion = document.getElementById('emotion');

// Clinical Information Fields
const diagnosis = document.getElementById('diagnosis');
const medication = document.getElementById('medication');
const therapy_type = document.getElementById('therapy_type');
const treatment_duration = document.getElementById('treatment_duration');
const treatment_progressSlider = document.getElementById('treatment_progress');
const treatment_progressValue = document.getElementById('treatment_progressValue');
const physical_activity = document.getElementById('physical_activity');

const resultModal = document.getElementById('resultModal');
const modalOverlay = document.getElementById('modalOverlay');
const modalCloseBtn = document.getElementById('modalCloseBtn');
const downloadPdfBtn = document.getElementById('downloadPdfBtn');

const historyContainer = document.getElementById('historyContainer');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

const errorMessage = document.getElementById('errorMessage');
const loadingSpinner = document.getElementById('loadingSpinner');

const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

let metricsChart = null;

// --- 3. INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    console.log('[OK] App initialized');
    initializeEventListeners();
    loadHistory();
    loadModelsInfo();
    updateSleepValue();
    updateTreatmentProgressValue();
});

function initializeEventListeners() {
    // Buttons
    analyzeBtn.addEventListener('click', runAnalysis);
    clearBtn.addEventListener('click', clearForm);
    modalCloseBtn.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', closeModal);
    downloadPdfBtn.addEventListener('click', downloadPDF);
    clearHistoryBtn.addEventListener('click', clearAllHistory);

    // Tabs
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => switchTab(e.target.dataset.tab));
    });

    // Sleep slider
    sleepSlider.addEventListener('input', updateSleepValue);

    // New sliders for mental health metrics
    stressSlider.addEventListener('input', updateStressValue);
    severitySlider.addEventListener('input', updateSeverityValue);
    mood_scoreSlider.addEventListener('input', updateMoodValue);
    treatment_progressSlider.addEventListener('input', updateTreatmentProgressValue);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

// --- 4. TAB SWITCHING ---
function switchTab(tabName) {
    // Update buttons
    tabBtns.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Update content
    tabContents.forEach(content => content.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
}

// --- 5. FORM MANAGEMENT ---
function updateSleepValue() {
    sleepValue.textContent = `${sleepSlider.value} hrs`;
}

function updateStressValue() {
    stressValue.textContent = stressSlider.value;
}

function updateSeverityValue() {
    severityValue.textContent = severitySlider.value;
}

function updateMoodValue() {
    mood_scoreValue.textContent = mood_scoreSlider.value;
}

function updateTreatmentProgressValue() {
    treatment_progressValue.textContent = `${treatment_progressSlider.value}%`;
}

function clearForm() {
    userInput.value = '';
    age.value = 25;
    gender.value = 'Male';
    height.value = 170;
    weight.value = 70;
    sleepSlider.value = 7;
    stressSlider.value = 5;
    severitySlider.value = 5;
    mood_scoreSlider.value = 5;
    emotion.value = 'Neutral';
    lifestyle.value = 'Sedentary';
    diagnosis.value = 'None';
    medication.value = 'None';
    therapy_type.value = 'None';
    treatment_duration.value = 0;
    treatment_progressSlider.value = 50;
    physical_activity.value = 0;
    errorMessage.style.display = 'none';
    updateSleepValue();
    updateStressValue();
    updateSeverityValue();
    updateMoodValue();
    closeModal();
}

// --- 6. ANALYSIS ---
async function runAnalysis() {
    // Clear previous errors
    errorMessage.style.display = 'none';

    // Validate input
    if (!userInput.value.trim()) {
        showError('Please enter your thoughts/story before running analysis.');
        return;
    }

    // Show loading
    loadingSpinner.style.display = 'block';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_input: userInput.value,
                age: parseInt(age.value),
                gender: gender.value,
                height: parseInt(height.value),
                weight: parseInt(weight.value),
                sleep: parseInt(sleepSlider.value),
                stress: parseInt(stressSlider.value),
                severity: parseInt(severitySlider.value),
                mood_score: parseInt(mood_scoreSlider.value),
                emotion: emotion.value,
                lifestyle: lifestyle.value,
                diagnosis: document.getElementById('diagnosis').value,
                medication: document.getElementById('medication').value,
                therapy_type: document.getElementById('therapy_type').value,
                treatment_duration: parseInt(document.getElementById('treatment_duration').value),
                treatment_progress: parseInt(document.getElementById('treatment_progress').value),
                physical_activity: parseFloat(document.getElementById('physical_activity').value),
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        currentResult = await response.json();
        console.log('[OK] Analysis complete:', currentResult);

        // Update history
        await loadHistory();

        // Show modal
        displayResult(currentResult);
        openModal();

    } catch (error) {
        console.error('[ERROR]', error);
        showError(`Error: ${error.message}`);
    } finally {
        loadingSpinner.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// --- 7. MODAL MANAGEMENT ---
function openModal() {
    resultModal.showModal();
    modalOverlay.style.display = 'block';
}

function closeModal() {
    resultModal.close();
    modalOverlay.style.display = 'none';
}

function displayResult(result) {
    // Risk level
    const riskLevel = document.getElementById('riskLevel');
    riskLevel.textContent = result.risk_level;
    riskLevel.className = `risk-level ${result.risk_level.toLowerCase()}`;

    // Confidence
    document.getElementById('confidenceValue').textContent = `${result.confidence.toFixed(1)}%`;

    // Recommendation
    document.getElementById('recommendationText').textContent = result.recommendation;

    // Demographics
    document.getElementById('displayAge').textContent = `${result.age} years`;
    document.getElementById('displayGender').textContent = result.gender;
    document.getElementById('displaySleep').textContent = `${result.sleep} hrs`;

    // Chart
    drawChart(result);
}

function drawChart(result) {
    const canvas = document.getElementById('metricsChart');
    const ctx = canvas.getContext('2d');

    // Clear previous chart
    if (metricsChart) {
        metricsChart.destroy();
    }

    // Prepare data
    const metrics = ['Risk Probability', 'Sleep Health', 'Activity Level'];
    const values = [
        result.confidence,
        result.sleep * 10,
        result.lifestyle === 'Active' ? 80 : (result.lifestyle === 'Sedentary' ? 20 : 50)
    ];
    const colors = ['#cc0000', '#4CAF50', '#2196F3'];

    // Draw bars manually (simple canvas approach)
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const barWidth = canvas.width / metrics.length / 1.5;
    const spacing = canvas.width / metrics.length;
    const maxHeight = canvas.height * 0.8;
    const scale = maxHeight / 100;

    ctx.fillStyle = '#333333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';

    // Draw bars
    for (let i = 0; i < metrics.length; i++) {
        const x = i * spacing + (spacing - barWidth) / 2;
        const height = (values[i] / 100) * maxHeight;
        const y = canvas.height - height - 40;

        // Bar
        ctx.fillStyle = colors[i];
        ctx.fillRect(x, y, barWidth, height);

        // Value label on bar
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Arial';
        ctx.fillText(values[i].toFixed(1), x + barWidth / 2, y - 5);

        // X-axis label
        ctx.fillStyle = '#cccccc';
        ctx.font = '12px Arial';
        const labelX = x + barWidth / 2;
        const labelY = canvas.height - 10;
        ctx.fillText(metrics[i], labelX, labelY);
    }

    // Y-axis
    ctx.strokeStyle = '#555555';
    ctx.beginPath();
    ctx.moveTo(30, 10);
    ctx.lineTo(30, canvas.height - 40);
    ctx.lineTo(canvas.width, canvas.height - 40);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = '#999999';
    ctx.font = '11px Arial';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i += 2) {
        const y = canvas.height - 40 - (i / 10) * maxHeight;
        ctx.fillText(i * 10, 25, y + 4);
    }
}

// --- 8. PDF DOWNLOAD ---
async function downloadPDF() {
    if (!currentResult) return;

    downloadPdfBtn.disabled = true;
    downloadPdfBtn.textContent = '⏳ Generating PDF...';

    try {
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentResult)
        });

        if (!response.ok) {
            throw new Error('PDF generation failed');
        }

        // Get blob
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `MindGuard_Report_${currentResult.timestamp.replace(/ /g, '_').replace(/:/g, '-')}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        console.log('[OK] PDF downloaded');
    } catch (error) {
        console.error('[ERROR]', error);
        showError(`PDF download failed: ${error.message}`);
    } finally {
        downloadPdfBtn.disabled = false;
        downloadPdfBtn.textContent = '📄 Download as PDF';
    }
}

// --- 9. HISTORY MANAGEMENT ---
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        analysisHistory = data.history;

        displayHistory();
    } catch (error) {
        console.error('[ERROR] Failed to load history:', error);
    }
}

function displayHistory() {
    if (analysisHistory.length === 0) {
        historyContainer.innerHTML = '<div class="history-empty"><p>No analyses yet. Run an analysis to see it here!</p></div>';
        clearHistoryBtn.style.display = 'none';
        return;
    }

    clearHistoryBtn.style.display = 'block';

    historyContainer.innerHTML = analysisHistory.map((result, index) => `
        <div class="history-item" data-index="${index}">
            <div class="history-item-top">
                <span class="history-item-risk ${result.risk_level.toLowerCase()}">
                    ${result.risk_level}
                </span>
                <span class="history-item-confidence">${result.confidence.toFixed(1)}%</span>
                <button class="history-item-delete" data-index="${index}" title="Delete this result">🗑️</button>
            </div>
            <div class="history-item-timestamp">${result.timestamp}</div>
        </div>
    `).join('');

    // Add event listeners
    document.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', (e) => {
            // Don't open if delete button clicked
            if (e.target.classList.contains('history-item-delete')) return;
            const index = item.dataset.index;
            currentResult = analysisHistory[index];
            displayResult(currentResult);
            openModal();
        });
    });

    document.querySelectorAll('.history-item-delete').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const index = btn.dataset.index;
            deleteHistoryItem(index);
        });
    });
}

async function deleteHistoryItem(index) {
    try {
        const response = await fetch(`/api/delete-history/${index}`, {
            method: 'POST'
        });

        if (response.ok) {
            await loadHistory();
            closeModal();
        }
    } catch (error) {
        console.error('[ERROR]', error);
        showError('Failed to delete history item');
    }
}

async function clearAllHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;

    try {
        const response = await fetch('/api/clear-history', {
            method: 'POST'
        });

        if (response.ok) {
            await loadHistory();
            closeModal();
        }
    } catch (error) {
        console.error('[ERROR]', error);
        showError('Failed to clear history');
    }
}

// --- 10. MODELS INFO ---
async function loadModelsInfo() {
    try {
        const response = await fetch('/api/models-info');
        const data = await response.json();
        document.getElementById('reportContent').textContent = data.report;
    } catch (error) {
        console.error('[ERROR]', error);
        document.getElementById('reportContent').textContent = 'Failed to load report';
    }
}

// --- 11. ERROR HANDLING ---
function showError(message) {
    errorMessage.textContent = `❌ ${message}`;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

console.log('[OK] JavaScript loaded');
