// DOM Elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-chat');
const reloadBtn = document.getElementById('reload-model');
const micBtn = document.getElementById('mic-btn');
const toggleVoiceBtn = document.getElementById('toggle-voice-out');
const welcomeMessage = document.querySelector('.welcome-message');
const noticeModal = document.getElementById('notice-modal');
const closeModalBtn = document.getElementById('close-modal');
const connectionStatus = document.getElementById('connection-status');
const mockBadge = document.getElementById('mock-badge');

// State
let conversationHistory = [];
let isGenerating = false;
let isVoiceOutputEnabled = true;

// Audio Recording State
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Toggle Voice Output
toggleVoiceBtn.addEventListener('click', () => {
    isVoiceOutputEnabled = !isVoiceOutputEnabled;
    toggleVoiceBtn.classList.toggle('active', isVoiceOutputEnabled);
    const icon = toggleVoiceBtn.querySelector('i');
    if (isVoiceOutputEnabled) {
        icon.className = 'ph-fill ph-speaker-high';
    } else {
        icon.className = 'ph ph-speaker-slash';
    }
});

// Microphone Handling
micBtn.addEventListener('mousedown', startRecording);
micBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });

micBtn.addEventListener('mouseup', stopRecording);
micBtn.addEventListener('mouseleave', stopRecording);
micBtn.addEventListener('touchend', stopRecording);

async function startRecording() {
    if (isRecording || isGenerating) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = processAudio;
        mediaRecorder.start();

        isRecording = true;
        micBtn.classList.add('recording');
        messageInput.placeholder = "듣고 있습니다...";

    } catch (err) {
        console.error("Microphone access denied:", err);
        alert("마이크 접근 권한이 필요합니다.");
    }
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    micBtn.classList.remove('recording');
    messageInput.placeholder = "음성을 처리하는 중...";
}

async function processAudio() {
    if (audioChunks.length === 0) return;

    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
        const response = await fetch('/api/stt', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            console.error("STT Error:", data.error);
            alert("음성 인식 오류: " + data.error);
        } else if (data.text) {
            messageInput.value = data.text;
            // Update input height
            messageInput.style.height = 'auto';
            messageInput.style.height = (messageInput.scrollHeight) + 'px';
            sendBtn.disabled = false;
        }
    } catch (error) {
        console.error("Error sending audio to STT:", error);
    } finally {
        messageInput.placeholder = "메시지 입력 (음성: 마이크 클릭, 전송: Enter)";
    }
}

// Auto-resize textarea
messageInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';

    // Enable/disable send button
    if (this.value.trim().length > 0 && !isGenerating) {
        sendBtn.disabled = false;
    } else {
        sendBtn.disabled = true;
    }
});

// Handle Enter key (Submit on Enter, newline on Shift+Enter)
messageInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) {
            sendMessage();
        }
    }
});

// Handle send button click
sendBtn.addEventListener('click', sendMessage);

// Handle clear button
clearBtn.addEventListener('click', () => {
    // Keep only welcome message, remove all others
    Array.from(chatContainer.children).forEach(child => {
        if (!child.classList.contains('welcome-message')) {
            child.remove();
        }
    });
    welcomeMessage.style.display = 'flex';
    conversationHistory = [];
});

// Reload Model logic
reloadBtn.addEventListener('click', async () => {
    reloadBtn.classList.add('loading');
    const icon = reloadBtn.querySelector('i');
    icon.className = 'ph ph-circle-notch loading';

    try {
        const response = await fetch('/api/reload', { method: 'POST' });
        const data = await response.json();
        if (data.status === "success") {
            console.log("Brain reloaded!");
            // Temporary success indicator
            reloadBtn.style.color = "#22c55e";
            setTimeout(() => {
                reloadBtn.style.color = "";
                icon.className = 'ph ph-arrows-clockwise';
                reloadBtn.classList.remove('loading');
            }, 1000);
        }
    } catch (error) {
        console.error("Reload failed:", error);
        reloadBtn.style.color = "#ef4444";
        setTimeout(() => {
            reloadBtn.style.color = "";
            icon.className = 'ph ph-arrows-clockwise';
            reloadBtn.classList.remove('loading');
        }, 1000);
    }
});

closeModalBtn.addEventListener('click', () => {
    noticeModal.classList.add('hidden');
});

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function appendUserMessage(text) {
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    const row = document.createElement('div');
    row.className = 'message-row user';

    row.innerHTML = `
        <div class="message-bubble">
            <div class="message-content"></div>
        </div>
    `;

    // Safely set text content
    row.querySelector('.message-content').textContent = text;
    chatContainer.appendChild(row);
    scrollToBottom();
}

function createModelMessagePlaceholder() {
    const row = document.createElement('div');
    row.className = 'message-row model';

    row.innerHTML = `
        <div class="message-bubble">
            <div class="model-avatar">
                <i class="ph-fill ph-robot"></i>
            </div>
            <div class="message-content"><span class="cursor-blink"></span></div>
        </div>
    `;

    chatContainer.appendChild(row);
    scrollToBottom();

    // Return the content div where text will be streamed
    return {
        row: row,
        contentNode: row.querySelector('.message-content')
    };
}

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isGenerating) return;

    // Reset input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;
    isGenerating = true;

    // Add user message to UI and history
    appendUserMessage(text);
    conversationHistory.push({ role: "user", content: text });

    // Create placeholder for model response
    const { contentNode } = createModelMessagePlaceholder();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: conversationHistory,
                temperature: 0.7,
                top_k: 50,
                max_new_tokens: 300
            })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        // Handle Server-Sent Events (SSE) streaming
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        let assistantResponse = "";
        let isDone = false;

        // Strip cursor for writing
        contentNode.innerHTML = '<span class="cursor-blink"></span>';

        while (!isDone) {
            const { value, done: streamDone } = await reader.read();
            if (streamDone) break;

            const chunk = decoder.decode(value, { stream: true });

            // SSE chunks look like "data: {...}\n\n"
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.replace('data: ', '').trim();
                    if (!jsonStr) continue;

                    try {
                        const data = JSON.parse(jsonStr);
                        assistantResponse = data.text;

                        // Update UI seamlessly with Thought parsing
                        renderStreamingMessage(contentNode, assistantResponse);
                        scrollToBottom();

                        if (data.done) {
                            isDone = true;
                        }
                    } catch (e) {
                        console.error('Error parsing streaming JSON:', e, jsonStr);
                    }
                }
            }
        }

        // Clean up cursor and save history
        renderStreamingMessage(contentNode, assistantResponse, true); // Final render without cursor
        conversationHistory.push({ role: "assistant", content: assistantResponse });

        // Play TTS Voice if enabled
        if (isVoiceOutputEnabled && assistantResponse) {
            playTTS(assistantResponse);
        }

    } catch (error) {
        console.error('Chat error:', error);
        contentNode.textContent = "오류가 발생했습니다. 서버가 실행 중인지 확인해주세요.";
        contentNode.style.color = "#ef4444";
        noticeModal.classList.remove('hidden');

        // Remove from history so it doesn't break future context
        conversationHistory.pop();
    } finally {
        isGenerating = false;
        // Re-evaluate send button state
        if (messageInput.value.trim().length > 0) {
            sendBtn.disabled = false;
        }
    }
}

/**
 * Parses streaming text for <|thought|> and <|assistant|> tags
 * and renders them into the UI with premium styling.
 */
function renderStreamingMessage(node, fullText, isFinal = false) {
    node.innerHTML = ''; // Clear existing

    const thoughtMarker = "<|thought|>";
    const assistantMarker = "<|assistant|>";

    let workingText = fullText;

    // 1. Extract thought if present
    if (workingText.includes(thoughtMarker)) {
        const parts = workingText.split(thoughtMarker);
        // Any text BEFORE <|thought|> is usually system or user echo, we skip it
        const contentAfterThoughtStart = parts[1] || "";

        if (contentAfterThoughtStart.includes(assistantMarker)) {
            const subParts = contentAfterThoughtStart.split(assistantMarker);
            const thoughtContent = subParts[0].trim();
            const assistantContent = subParts[1] || "";

            if (thoughtContent) {
                const thoughtDiv = document.createElement('div');
                thoughtDiv.className = 'thought-block';
                thoughtDiv.textContent = thoughtContent;
                node.appendChild(thoughtDiv);
            }

            const textNode = document.createTextNode(assistantContent);
            node.appendChild(textNode);
        } else {
            // Still thinking...
            const thoughtDiv = document.createElement('div');
            thoughtDiv.className = 'thought-block';
            thoughtDiv.textContent = contentAfterThoughtStart.trim();
            node.appendChild(thoughtDiv);
        }
    } else if (workingText.includes(assistantMarker)) {
        // Just assistant text
        const content = workingText.split(assistantMarker)[1] || "";
        node.appendChild(document.createTextNode(content));
    } else {
        // Fallback or plain text
        node.appendChild(document.createTextNode(workingText));
    }

    // Add cursor during generation
    if (!isFinal) {
        const cursor = document.createElement('span');
        cursor.className = 'cursor-blink';
        node.appendChild(cursor);
    }
}

async function playTTS(text) {
    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            const errData = await response.json();
            console.error("TTS Error:", errData.error);
            return;
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);

        // Add visual indicator that model is speaking
        const robotIcon = document.querySelector('.header-content i.ph-robot');
        robotIcon.classList.add('pulse');

        audio.onended = () => {
            URL.revokeObjectURL(url);
            robotIcon.classList.remove('pulse');
        };

        audio.play();

    } catch (error) {
        console.error("Failed to play TTS audio:", error);
    }
}

// Check server status on load
async function checkServerStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (data.status === "online") {
            connectionStatus.textContent = "Online";
            if (data.mock_mode) {
                mockBadge.classList.remove('hidden');
            } else {
                mockBadge.classList.add('hidden');
            }
        }
    } catch (error) {
        console.error("Server offline or status check failed:", error);
        connectionStatus.textContent = "Offline";
    }
}

checkServerStatus();
