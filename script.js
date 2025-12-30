const askBtn = document.getElementById('askBtn');
const userQuery = document.getElementById('userQuery');
const loadingIndicator = document.getElementById('loadingIndicator');
const responseContainer = document.getElementById('responseContainer');
const summaryText = document.getElementById('summaryText');
const excerptsList = document.getElementById('excerptsList');

// Audio Controls
const playBtn = document.getElementById('playBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const audioStatus = document.getElementById('audioStatus');

let currentAudio = new Audio();
let currentTextToRead = "";
let isAudioLoading = false;

// 1. Fill Query
function fillQuery(text) {
    userQuery.value = text;
}

// 2. Search Logic
async function handleSearch() {
    const query = userQuery.value.trim();
    if (!query) return;

    // Reset UI
    loadingIndicator.classList.remove('hidden');
    responseContainer.classList.add('hidden');
    
    // Reset Audio completely on new search
    stopAndResetAudio(); 

    try {
        const response = await fetch('http://127.0.0.1:8000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query })
        });

        const data = await response.json();

        if (response.ok) {
            // Update Text
            summaryText.innerText = data.summary;
            currentTextToRead = data.summary; // Save for audio

            // Update Excerpts
            excerptsList.innerHTML = '';
            data.excerpts.forEach(excerpt => {
                const div = document.createElement('div');
                div.className = "bg-[#f4f1ea] p-4 border-l-4 border-[#4b5320] text-sm italic text-gray-700 relative";
                div.innerHTML = `
                    <p class="mb-2">"${excerpt.text}"</p>
                    <span class="text-xs text-gray-500 not-italic block text-right">— ${excerpt.chapter}</span>
                `;
                excerptsList.appendChild(div);
            });

            loadingIndicator.classList.add('hidden');
            responseContainer.classList.remove('hidden');
        } else {
            alert("Error retrieving archives.");
            loadingIndicator.classList.add('hidden');
        }

    } catch (error) {
        console.error(error);
        alert("Server error. Ensure server.py is running.");
        loadingIndicator.classList.add('hidden');
    }
}

// 3. Audio Logic

async function playAudio() {
    // If playing, do nothing
    if (!currentAudio.paused && currentAudio.currentTime > 0) return;
    
    // If audio is loaded (paused), just resume
    if (currentAudio.src && currentAudio.src !== "") {
        currentAudio.play();
        audioStatus.innerText = "Playing...";
        return;
    }

    // New Audio Request
    if (!currentTextToRead) return;

    try {
        isAudioLoading = true;
        audioStatus.innerText = "Generating Voice...";
        playBtn.disabled = true;

        const response = await fetch('http://127.0.0.1:8000/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentTextToRead })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            currentAudio.src = url;
            currentAudio.play();
            audioStatus.innerText = "Playing...";
        } else {
            audioStatus.innerText = "Audio Error";
        }
    } catch (e) {
        console.error(e);
        audioStatus.innerText = "Connection Error";
    } finally {
        isAudioLoading = false;
        playBtn.disabled = false;
    }
}

function pauseAudio() {
    if (!currentAudio.paused) {
        currentAudio.pause();
        audioStatus.innerText = "Paused";
    }
}

function stopAndResetAudio() {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    // We DON'T clear src here if we want "Reset" to just mean "Go back to start"
    // If you want Reset to mean "Clear buffer", uncomment next line:
    // currentAudio.src = ""; 
    audioStatus.innerText = "Reset";
}

// Event Listeners
askBtn.addEventListener('click', handleSearch);
userQuery.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSearch();
    }
});

playBtn.addEventListener('click', playAudio);
pauseBtn.addEventListener('click', pauseAudio);
resetBtn.addEventListener('click', stopAndResetAudio);

currentAudio.onended = () => {
    audioStatus.innerText = "Finished";
};