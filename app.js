/*
 SignSpeak - real hand-gesture recognition (no random predictions)
 -----------------------------------------------------------------
 This file detects up to two hands using MediaPipe Hands.
 It reliably recognizes a small set of geometric gestures:
 - STOP / 5        : open palm
 - PEACE / 2       : index and middle fingers extended
 - ONE / I         : index finger extended
 - YES / FIST      : closed fist
 - THANK YOU       : two open hands held close together

 IMPORTANT:
 It intentionally does NOT pretend to recognize every dictionary word.
 For all remaining signs, it displays “Sign not recognized”.
 To recognize your complete dictionary accurately, train a model and set
 MODEL_URL and LABELS_URL below. Instructions are included at the bottom.
*/

let videoElement;
let canvasElement;
let canvasCtx;
let hands;
let camera;
let isCameraRunning = false;
let currentFacingMode = 'user';
let speechVoices = [];
let translatedWords = [];
let signHistory = [];
let currentMode = 'word';
let model = null;
let modelLabels = [];
let modelReady = false;
let processingModel = false;
let lastCommittedLabel = '';
let lastCommitTime = 0;
let candidateBuffer = [];

const MODEL_URL = 'model/model.json';
const LABELS_URL = 'model/labels.json';
const MIN_CONFIDENCE = 0.85;
const STABLE_FRAMES = 8;
const COMMIT_COOLDOWN_MS = 1600;

const SIGN_VOCABULARY = [
    { id: 'stop', name: 'Stop', emoji: '✋', hands: 1, category: 'actions', description: 'Open palm facing the camera' },
    { id: 'peace', name: 'Peace', emoji: '✌️', hands: 1, category: 'greetings', description: 'Index and middle fingers extended' },
    { id: 'i', name: 'I', emoji: '👆', hands: 1, category: 'basic', description: 'Index finger extended' },
    { id: 'yes', name: 'Yes', emoji: '✊', hands: 1, category: 'basic', description: 'Closed fist' },
    { id: 'thank_you', name: 'Thank You', emoji: '🙏', hands: 2, category: 'greetings', description: 'Two open hands held close together' },
    { id: 'hello', name: 'Hello', emoji: '👋', hands: 1, category: 'greetings', description: 'Use a trained model for the moving wave gesture' },
    { id: 'please', name: 'Please', emoji: '🤲', hands: 2, category: 'greetings', description: 'Use a trained model for accurate recognition' },
    { id: 'help', name: 'Help', emoji: '🆘', hands: 2, category: 'actions', description: 'Use a trained model for accurate recognition' },
    { id: 'water', name: 'Water', emoji: '💧', hands: 1, category: 'basic', description: 'Use a trained model for accurate recognition' },
    { id: 'food', name: 'Food', emoji: '🍽️', hands: 1, category: 'basic', description: 'Use a trained model for accurate recognition' }
];

const ALPHABET_SIGNS = [
    { id: 'A', name: 'A', emoji: '🅰️', hands: 1 },
    { id: 'B', name: 'B', emoji: '🅱️', hands: 1 },
    { id: 'I', name: 'I', emoji: '🇮', hands: 1 },
    { id: 'L', name: 'L', emoji: '🇱', hands: 1 },
    { id: 'V', name: 'V', emoji: '✌️', hands: 1 }
];

const NUMBER_SIGNS = [
    { id: '0', name: '0', emoji: '0️⃣', hands: 1 },
    { id: '1', name: '1', emoji: '1️⃣', hands: 1 },
    { id: '2', name: '2', emoji: '2️⃣', hands: 1 },
    { id: '5', name: '5', emoji: '5️⃣', hands: 1 }
];

window.addEventListener('DOMContentLoaded', async () => {
    videoElement = document.getElementById('webcam');
    canvasElement = document.getElementById('output_canvas');
    canvasCtx = canvasElement.getContext('2d');

    bindButtons();
    renderDictionary();
    loadSpeechVoices();
    setMode('word');
    showStatus('Camera is off. Click Start Camera.', 'info');

    if (window.speechSynthesis) {
        speechSynthesis.onvoiceschanged = loadSpeechVoices;
    }

    if (window.tf) {
        await loadTrainedModel();
    }
});

function bindButtons() {
    byId('startCamera')?.addEventListener('click', startCamera);
    byId('stopCamera')?.addEventListener('click', stopCamera);
    byId('switchCamera')?.addEventListener('click', switchCamera);
    byId('clear-text')?.addEventListener('click', clearText);
    byId('delete-last')?.addEventListener('click', deleteLastWord);
    byId('add-space')?.addEventListener('click', addSpace);
    byId('speak-btn')?.addEventListener('click', speakText);
    byId('stop-speak')?.addEventListener('click', () => speechSynthesis.cancel());
    byId('mode-word')?.addEventListener('click', () => setMode('word'));
    byId('mode-alphabet')?.addEventListener('click', () => setMode('alphabet'));
    byId('mode-number')?.addEventListener('click', () => setMode('number'));
    byId('sign-search')?.addEventListener('input', renderDictionary);
    byId('category-filter')?.addEventListener('change', renderDictionary);
    byId('rate-slider')?.addEventListener('input', updateVoiceLabels);
    byId('pitch-slider')?.addEventListener('input', updateVoiceLabels);
}

function byId(id) {
    return document.getElementById(id);
}

function initializeHands() {
    hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.75,
        minTrackingConfidence: 0.70
    });

    hands.onResults(onHandsResults);
}

async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        alert('This browser does not support camera access. Use a modern Chrome, Edge, or Firefox browser.');
        return;
    }

    try {
        byId('loading-indicator')?.classList.remove('hidden');

        if (!hands) initializeHands();

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        videoElement.srcObject = stream;
        await videoElement.play();

        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;

        camera = new Camera(videoElement, {
            onFrame: async () => {
                if (hands) await hands.send({ image: videoElement });
            },
            width: videoElement.videoWidth,
            height: videoElement.videoHeight
        });

        await camera.start();
        isCameraRunning = true;
        toggleCameraButtons(true);
        byId('loading-indicator')?.classList.add('hidden');
        showStatus('Camera started. Show one or two hands clearly.', 'info');
    } catch (error) {
        console.error(error);
        byId('loading-indicator')?.classList.add('hidden');
        alert('Camera could not start. Run the project on localhost or HTTPS, then allow camera permission.');
    }
}

function stopCamera() {
    if (camera) {
        camera.stop();
        camera = null;
    }

    const stream = videoElement?.srcObject;
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (videoElement) videoElement.srcObject = null;

    isCameraRunning = false;
    candidateBuffer = [];
    lastCommittedLabel = '';
    canvasCtx?.clearRect(0, 0, canvasElement.width, canvasElement.height);
    toggleCameraButtons(false);
    showStatus('Camera stopped.', 'info');
}

async function switchCamera() {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    if (!isCameraRunning) return;
    stopCamera();
    await new Promise(resolve => setTimeout(resolve, 300));
    await startCamera();
}

function toggleCameraButtons(running) {
    if (byId('startCamera')) byId('startCamera').disabled = running;
    if (byId('stopCamera')) byId('stopCamera').disabled = !running;
    if (byId('switchCamera')) byId('switchCamera').disabled = !running;
}

async function onHandsResults(results) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    const landmarksList = results.multiHandLandmarks || [];
    const handednessList = results.multiHandedness || [];

    if (landmarksList.length === 0) {
        candidateBuffer = [];
        showStatus('Show your hand(s) to the camera.', 'info');
        return;
    }

    const detectedHands = landmarksList.map((landmarks, index) => {
        const label = handednessList[index]?.label || `Hand ${index + 1}`;
        drawHand(landmarks, label, index);
        return { landmarks, label };
    });

    showStatus(`${detectedHands.length} hand${detectedHands.length === 1 ? '' : 's'} detected. Hold the sign still.`, 'info');

    let prediction = null;
    if (modelReady && !processingModel) {
        prediction = await predictWithModel(detectedHands);
    } else {
        prediction = classifySupportedGeometricSigns(detectedHands);
    }

    const stablePrediction = getStablePrediction(prediction);

    if (!stablePrediction) {
        if (prediction) {
            displayPrediction(prediction, false);
        } else {
            displayUnknown(detectedHands.length);
        }
        return;
    }

    displayPrediction(stablePrediction, true);
    commitPrediction(stablePrediction);
}

function drawHand(landmarks, label, index) {
    const connectorColor = index === 0 ? '#00e676' : '#ff3dcb';
    const pointColor = label === 'Left' ? '#ff5252' : '#40c4ff';

    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: connectorColor,
        lineWidth: 3
    });

    drawLandmarks(canvasCtx, landmarks, {
        color: pointColor,
        lineWidth: 1,
        radius: 4
    });

    const wrist = landmarks[0];
    canvasCtx.font = 'bold 16px Arial';
    canvasCtx.fillStyle = connectorColor;
    canvasCtx.fillText(label, wrist.x * canvasElement.width, wrist.y * canvasElement.height - 12);
}

function getStablePrediction(prediction) {
    if (!prediction || prediction.confidence < MIN_CONFIDENCE) {
        candidateBuffer = [];
        return null;
    }

    candidateBuffer.push(prediction.id);
    if (candidateBuffer.length > STABLE_FRAMES) candidateBuffer.shift();

    const stable = candidateBuffer.length === STABLE_FRAMES &&
        candidateBuffer.every(id => id === prediction.id);

    return stable ? prediction : null;
}

function commitPrediction(prediction) {
    const now = Date.now();
    const sameSignDuringCooldown = prediction.id === lastCommittedLabel &&
        now - lastCommitTime < COMMIT_COOLDOWN_MS;

    if (sameSignDuringCooldown) return;

    translatedWords.push(prediction.name);
    signHistory.push(prediction.name);
    lastCommittedLabel = prediction.id;
    lastCommitTime = now;

    byId('translated-text').value = translatedWords.join(' ');
    renderHistory();
    updateStatistics(prediction.confidence);
}

function classifySupportedGeometricSigns(detectedHands) {
    if (currentMode === 'alphabet') return classifyAlphabet(detectedHands);
    if (currentMode === 'number') return classifyNumbers(detectedHands);

    if (detectedHands.length === 2) {
        return classifyTwoHands(detectedHands[0].landmarks, detectedHands[1].landmarks);
    }

    return classifyOneHand(detectedHands[0].landmarks);
}

function classifyOneHand(landmarks) {
    const fingers = fingerStates(landmarks);
    const extended = Object.values(fingers).filter(Boolean).length;

    const onlyIndex = fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky;
    const indexAndMiddle = fingers.index && fingers.middle && !fingers.ring && !fingers.pinky;
    const openPalm = extended >= 4;
    const closedFist = extended === 0;

    if (openPalm) return prediction('stop', 'Stop', '✋', 0.92, 1);
    if (indexAndMiddle) return prediction('peace', 'Peace', '✌️', 0.91, 1);
    if (onlyIndex) return prediction('i', 'I', '👆', 0.88, 1);
    if (closedFist) return prediction('yes', 'Yes', '✊', 0.86, 1);

    return null;
}

function classifyTwoHands(handA, handB) {
    const leftOpen = isOpenPalm(handA);
    const rightOpen = isOpenPalm(handB);
    const wristDistance = distance(handA[0], handB[0]);
    const palmDistance = distance(handA[9], handB[9]);

    if (leftOpen && rightOpen && wristDistance < 0.32 && palmDistance < 0.25) {
        return prediction('thank_you', 'Thank You', '🙏', 0.90, 2);
    }

    return null;
}

function classifyAlphabet(detectedHands) {
    if (detectedHands.length !== 1) return null;
    const landmarks = detectedHands[0].landmarks;
    const fingers = fingerStates(landmarks);
    const extended = Object.values(fingers).filter(Boolean).length;

    const onlyIndex = fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky;
    const indexAndMiddle = fingers.index && fingers.middle && !fingers.ring && !fingers.pinky;
    const openPalm = extended >= 4;
    const fist = extended === 0;

    if (fist) return prediction('A', 'A', '🅰️', 0.85, 1);
    if (openPalm) return prediction('B', 'B', '🅱️', 0.86, 1);
    if (onlyIndex) return prediction('I', 'I', '🇮', 0.86, 1);
    if (indexAndMiddle) return prediction('V', 'V', '✌️', 0.90, 1);

    return null;
}

function classifyNumbers(detectedHands) {
    if (detectedHands.length !== 1) return null;
    const landmarks = detectedHands[0].landmarks;
    const fingers = fingerStates(landmarks);
    const extended = Object.values(fingers).filter(Boolean).length;

    const onlyIndex = fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky;
    const indexAndMiddle = fingers.index && fingers.middle && !fingers.ring && !fingers.pinky;

    if (extended === 0) return prediction('0', '0', '0️⃣', 0.85, 1);
    if (onlyIndex) return prediction('1', '1', '1️⃣', 0.88, 1);
    if (indexAndMiddle) return prediction('2', '2', '2️⃣', 0.90, 1);
    if (extended >= 4) return prediction('5', '5', '5️⃣', 0.90, 1);

    return null;
}

function prediction(id, name, emoji, confidence, handsUsed) {
    return { id, name, emoji, confidence, handsUsed };
}

function fingerStates(landmarks) {
    return {
        thumb: isFingerExtended(landmarks, 4, 3, 2),
        index: isFingerExtended(landmarks, 8, 6, 5),
        middle: isFingerExtended(landmarks, 12, 10, 9),
        ring: isFingerExtended(landmarks, 16, 14, 13),
        pinky: isFingerExtended(landmarks, 20, 18, 17)
    };
}

function isFingerExtended(landmarks, tipIndex, pipIndex, mcpIndex) {
    const wrist = landmarks[0];
    const tip = distance(landmarks[tipIndex], wrist);
    const pip = distance(landmarks[pipIndex], wrist);
    const mcp = distance(landmarks[mcpIndex], wrist);
    return tip > pip * 1.10 && pip > mcp * 0.98;
}

function isOpenPalm(landmarks) {
    return Object.values(fingerStates(landmarks)).filter(Boolean).length >= 4;
}

function distance(pointA, pointB) {
    const dx = pointA.x - pointB.x;
    const dy = pointA.y - pointB.y;
    const dz = (pointA.z || 0) - (pointB.z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function displayPrediction(sign, stable) {
    const confidence = Math.round(sign.confidence * 100);
    const state = stable ? 'Recognized' : 'Checking stability';
    const handsText = `${sign.handsUsed} hand${sign.handsUsed === 1 ? '' : 's'} detected`;

    byId('detected-sign').innerHTML = `
        <div class="detected-result">
            <div class="detected-emoji">${sign.emoji}</div>
            <div class="detected-name">${sign.name}</div>
            <div class="detected-meta">${stable ? '✅' : '⏳'} ${state} · 🖐️ ${handsText}</div>
        </div>
    `;

    byId('confidence-meter')?.classList.remove('hidden');
    byId('confidence-fill').style.width = `${confidence}%`;
    byId('confidence-text').textContent = `${confidence}%`;
}

function displayUnknown(handCount) {
    byId('detected-sign').innerHTML = `
        <div class="detected-result">
            <div class="detected-emoji">🤔</div>
            <div class="detected-name">Sign not recognized</div>
            <div class="detected-meta">🖐️ ${handCount} hand${handCount === 1 ? '' : 's'} detected · Hold a supported sign still</div>
        </div>
    `;

    byId('confidence-meter')?.classList.add('hidden');
}

function showStatus(message) {
    const overlay = byId('no-hand-detected');
    if (!overlay) return;
    overlay.innerHTML = `<p>${message}</p>`;
    overlay.classList.remove('hidden');
}

function clearText() {
    translatedWords = [];
    signHistory = [];
    candidateBuffer = [];
    lastCommittedLabel = '';
    byId('translated-text').value = '';
    renderHistory();
    displayUnknown(0);
}

function deleteLastWord() {
    translatedWords.pop();
    signHistory.pop();
    byId('translated-text').value = translatedWords.join(' ');
    renderHistory();
}

function addSpace() {
    if (translatedWords.length > 0) {
        translatedWords.push('|');
        byId('translated-text').value = translatedWords.join(' ').replaceAll(' | ', '   ');
    }
}

function renderHistory() {
    const container = byId('sign-history');
    if (!container) return;

    if (!signHistory.length) {
        container.innerHTML = '<p class="placeholder">No recognized signs yet</p>';
        return;
    }

    container.innerHTML = signHistory
        .slice(-20)
        .map(name => `<span class="history-item">${name}</span>`)
        .join('');
}

function loadSpeechVoices() {
    if (!window.speechSynthesis) return;
    speechVoices = speechSynthesis.getVoices();
    const select = byId('voice-select');
    if (!select) return;

    select.innerHTML = '';
    speechVoices.forEach((voice, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${voice.name} (${voice.lang})`;
        if (voice.default) option.selected = true;
        select.appendChild(option);
    });
}

function updateVoiceLabels() {
    if (byId('rate-value')) byId('rate-value').textContent = `${byId('rate-slider').value}x`;
    if (byId('pitch-value')) byId('pitch-value').textContent = byId('pitch-slider').value;
}

function speakText() {
    const text = byId('translated-text').value.trim();
    if (!text) {
        alert('There is no recognized text to speak.');
        return;
    }

    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const selectedVoice = speechVoices[Number(byId('voice-select')?.value || 0)];
    if (selectedVoice) utterance.voice = selectedVoice;
    utterance.rate = Number(byId('rate-slider')?.value || 1);
    utterance.pitch = Number(byId('pitch-slider')?.value || 1);
    speechSynthesis.speak(utterance);
}

function setMode(mode) {
    currentMode = mode;
    candidateBuffer = [];
    lastCommittedLabel = '';

    document.querySelectorAll('.mode-btn').forEach(button => button.classList.remove('active'));
    byId(`mode-${mode}`)?.classList.add('active');

    const title = byId('current-mode-label');
    const description = byId('mode-description');
    if (!title || !description) return;

    if (mode === 'word') {
        title.textContent = '📝 Word Mode';
        description.textContent = modelReady ? 'Trained model recognition is active' : 'Built-in supported gestures: Stop, Peace, I, Yes, Thank You';
    }
    if (mode === 'alphabet') {
        title.textContent = '🔤 Alphabet Mode';
        description.textContent = 'Built-in supported letters: A, B, I, V';
    }
    if (mode === 'number') {
        title.textContent = '🔢 Number Mode';
        description.textContent = 'Built-in supported numbers: 0, 1, 2, 5';
    }
}

function renderDictionary() {
    const container = byId('sign-cards');
    if (!container) return;

    const search = (byId('sign-search')?.value || '').toLowerCase().trim();
    const category = byId('category-filter')?.value || 'all';

    const filtered = SIGN_VOCABULARY.filter(sign => {
        const matchesSearch = sign.name.toLowerCase().includes(search);
        const matchesCategory = category === 'all' || sign.category === category;
        return matchesSearch && matchesCategory;
    });

    container.innerHTML = filtered.map(sign => `
        <article class="sign-card">
            <h3>${sign.name}</h3>
            <div class="sign-emoji">${sign.emoji}</div>
            <span class="sign-category">${sign.category}</span>
            <span class="sign-category">${sign.hands === 2 ? '👐 Two hands' : '🖐️ One hand'}</span>
            <p class="sign-description">${sign.description}</p>
        </article>
    `).join('');
}

function updateStatistics(confidence) {
    const total = Number(localStorage.getItem('signSpeakTotalSigns') || 0) + 1;
    const confidenceTotal = Number(localStorage.getItem('signSpeakConfidenceSum') || 0) + confidence;
    localStorage.setItem('signSpeakTotalSigns', total);
    localStorage.setItem('signSpeakConfidenceSum', confidenceTotal);

    if (byId('stat-total-detected')) byId('stat-total-detected').textContent = total;
    if (byId('stat-today-count')) byId('stat-today-count').textContent = signHistory.length;
    if (byId('stat-avg-confidence')) byId('stat-avg-confidence').textContent = `${Math.round((confidenceTotal / total) * 100)}%`;
}

async function loadTrainedModel() {
    try {
        const modelResponse = await fetch(MODEL_URL, { method: 'HEAD' });
        if (!modelResponse.ok) return;

        model = await tf.loadLayersModel(MODEL_URL);
        const response = await fetch(LABELS_URL);
        modelLabels = await response.json();
        modelReady = true;
        console.log('Trained sign model loaded successfully.');
    } catch (error) {
        modelReady = false;
        console.info('No trained model found. Using the built-in limited gesture recognizer.');
    }
}

async function predictWithModel(detectedHands) {
    if (!modelReady || processingModel) return null;
    processingModel = true;

    try {
        const features = createModelInput(detectedHands);
        const input = tf.tensor2d([features], [1, 126]);
        const output = model.predict(input);
        const probabilities = await output.data();

        input.dispose();
        output.dispose();

        let bestIndex = 0;
        for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
        }

        const confidence = probabilities[bestIndex];
        if (confidence < MIN_CONFIDENCE) return null;

        const label = modelLabels[bestIndex];
        const dictionarySign = SIGN_VOCABULARY.find(item => item.id === label);
        if (!dictionarySign) return null;

        return {
            ...dictionarySign,
            confidence,
            handsUsed: detectedHands.length
        };
    } catch (error) {
        console.error('Prediction error:', error);
        return null;
    } finally {
        processingModel = false;
    }
}

function createModelInput(detectedHands) {
    const features = new Array(126).fill(0);
    const orderedHands = [...detectedHands].sort((a, b) => a.label.localeCompare(b.label));

    orderedHands.slice(0, 2).forEach((hand, handIndex) => {
        const wrist = hand.landmarks[0];
        hand.landmarks.forEach((point, landmarkIndex) => {
            const offset = handIndex * 63 + landmarkIndex * 3;
            features[offset] = point.x - wrist.x;
            features[offset + 1] = point.y - wrist.y;
            features[offset + 2] = point.z - wrist.z;
        });
    });

    return features;
}