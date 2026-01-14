const askBtn = document.getElementById('askBtn');
const userQuery = document.getElementById('userQuery');
const loadingIndicator = document.getElementById('loadingIndicator');
const responseContainer = document.getElementById('responseContainer');
const summaryText = document.getElementById('summaryText');
const excerptsList = document.getElementById('excerptsList');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// -------------------------------------------------------------
// CONFIGURATION: PRODUCTION vs LOCAL
// -------------------------------------------------------------
const RENDER_URL = "https://captain-jim.onrender.com"; 
const LOCAL_URL = "http://127.0.0.1:8000";

const API_BASE_URL = RENDER_URL;
// -------------------------------------------------------------

let currentAudio = new Audio();
let currentlyPlayingBtn = null; 
let currentResetBtn = null;     

// --- TOAST NOTIFICATION SYSTEM ---
function showToast(message, isError = true) {
    toastMessage.innerText = message;
    
    // Style: Red for error, Green/Gold for info
    if (isError) {
        toast.className = "fixed top-5 left-1/2 transform -translate-x-1/2 bg-red-900 text-white px-6 py-4 rounded shadow-2xl z-50 flex transition-all duration-300 w-[90%] max-w-md text-center border-2 border-[#fdf6e3]";
    } else {
        // Gold/Green for 'Info' or 'Loading' messages
        toast.className = "fixed top-5 left-1/2 transform -translate-x-1/2 bg-[#4b5320] text-white px-6 py-4 rounded shadow-2xl z-50 flex transition-all duration-300 w-[90%] max-w-md text-center border-2 border-[#fdf6e3]";
    }

    toast.classList.remove('hidden');

    // Hide automatically after 4 seconds
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 4000);
}

function fillQuery(text) {
    userQuery.value = text;
}

async function handleSearch() {
    const query = userQuery.value.trim();
    if (!query) return;

    loadingIndicator.classList.remove('hidden');
    responseContainer.classList.add('hidden');
    stopAudio(); 

    try {
        // Timeout Controller for Cold Starts (60 seconds)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); 

        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        // Handle Rate Limiting
        if (response.status === 429) {
            showToast("Whoa there! Too many questions too fast. Please wait a minute.");
            loadingIndicator.classList.add('hidden');
            return;
        }

        const data = await response.json();

        if (response.ok) {
            summaryText.innerText = data.summary;
            excerptsList.innerHTML = '';
            
            data.excerpts.forEach((excerpt, index) => {
                let sourceDisplay = excerpt.chapter;
                if (sourceDisplay.includes("Three_Day_Pass")) {
                    sourceDisplay = "From the memoir 'Three Day Pass'";
                } else {
                    sourceDisplay = sourceDisplay.replace(/_/g, ' ').replace('.txt', '');
                }

                const div = document.createElement('div');
                div.className = "bg-[#f4f1ea] p-4 border-l-4 border-[#4b5320] relative mb-4";
                div.innerHTML = `
                    <p class="mb-4 text-sm italic text-gray-700 leading-relaxed">"${excerpt.text}"</p>
                    <div class="flex justify-between items-end border-t border-gray-300 pt-3">
                        <span class="text-xs text-gray-500 font-bold uppercase tracking-wider">— ${sourceDisplay}</span>
                        <div class="flex items-center gap-2">
                            <button id="reset-btn-${index}" onclick="resetSpecificAudio(${index})"
                                class="hidden bg-gray-200 text-gray-600 px-3 py-1 text-xs uppercase tracking-wider hover:bg-red-100 hover:text-red-700 rounded transition-all">
                                ↻ Reset
                            </button>
                            <button id="play-btn-${index}" onclick="playExcerptAudio(this, '${escapeHtml(excerpt.text)}', ${index})" 
                                    class="bg-[#4b5320] text-white px-3 py-1 text-xs uppercase tracking-wider hover:bg-[#3a4119] flex items-center gap-2 rounded transition-all">
                                <span>▶ Hear Captain Jim</span>
                            </button>
                        </div>
                    </div>
                `;
                excerptsList.appendChild(div);
            });

            loadingIndicator.classList.add('hidden');
            responseContainer.classList.remove('hidden');
        } else {
            showToast("Archives unavailable. Status: " + response.status);
            loadingIndicator.classList.add('hidden');
        }

    } catch (error) {
        console.error(error);
        if (error.name === 'AbortError') {
            // Cold Start Message
            showToast("The server is waking up! Please click 'Ask Jim' one more time.", false);
        } else if (error.message.includes("Failed to fetch")) {
            showToast("Connection failed. Check your internet or disable AdBlockers.");
        } else {
            showToast("System Error: " + error.message);
        }
        loadingIndicator.classList.add('hidden');
    }
}

function escapeHtml(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

async function playExcerptAudio(btnElement, textToSpeak, index) {
    const relatedResetBtn = document.getElementById(`reset-btn-${index}`);

    // If currently playing, treat click as Pause/Resume
    if (currentlyPlayingBtn === btnElement && currentAudio.src) {
        if (currentAudio.paused) {
            currentAudio.play();
            btnElement.innerHTML = "<span>II Pause</span>";
            btnElement.classList.add('playing-audio');
        } else {
            currentAudio.pause();
            btnElement.innerHTML = "<span>▶ Resume</span>";
            btnElement.classList.remove('playing-audio');
        }
        return; 
    }

    stopAudio();

    // 1. Notify user we are starting the fetch
    showToast("Retrieving audio from archives... please wait.", false);

    // 2. Prime mobile audio driver
    currentAudio = new Audio(); 
    currentAudio.play().catch(e => {
        // Swallow the expected error from playing empty audio
    });

    // 3. UI Loading State
    const originalText = btnElement.innerHTML;
    btnElement.innerHTML = "<span>... Loading ...</span>";
    btnElement.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/speak`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textToSpeak })
        });

        if (response.ok) {
            const blob = await response.blob();
            
            // If blob is tiny (< 1KB), it's the JSON error message hidden in a blob
            if (blob.size < 1000) {
                console.warn("Audio blob too small, likely an API error message.");
                showToast("Captain Jim is on vocal rest (API Quota Limit Reached). Please try again next month!");
                resetButton(btnElement);
                return;
            }

            const url = URL.createObjectURL(blob);
            currentAudio.src = url;
            
            // 5. Play real audio
            currentAudio.play()
                .then(() => {
                    // Success!
                    currentlyPlayingBtn = btnElement;
                    currentResetBtn = relatedResetBtn;
                    btnElement.classList.add('playing-audio'); 
                    btnElement.innerHTML = "<span>II Pause</span>";
                    btnElement.disabled = false;
                    if (relatedResetBtn) relatedResetBtn.classList.remove('hidden');
                })
                .catch(e => {
                    console.error("Playback failed:", e);
                    showToast("Audio playback failed. Please try clicking again.");
                    resetButton(btnElement);
                });

            currentAudio.onended = () => {
                resetSpecificAudio(index);
            };

        } else {
            showToast("Audio Error: " + response.statusText);
            resetSpecificAudio(index);
        }
    } catch (e) {
        console.error(e);
        showToast("Connection Error while fetching audio.");
        resetSpecificAudio(index);
    }
}

function stopAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    if (currentlyPlayingBtn) {
        currentlyPlayingBtn.innerHTML = "<span>▶ Hear Captain Jim</span>";
        currentlyPlayingBtn.classList.remove('playing-audio');
        currentlyPlayingBtn.disabled = false;
        currentlyPlayingBtn = null;
    }
    if (currentResetBtn) {
        currentResetBtn.classList.add('hidden');
        currentResetBtn = null;
    }
}

function resetSpecificAudio(index) {
    const playBtn = document.getElementById(`play-btn-${index}`);
    const resetBtn = document.getElementById(`reset-btn-${index}`);

    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }

    if (playBtn) {
        playBtn.innerHTML = "<span>▶ Hear Captain Jim</span>";
        playBtn.classList.remove('playing-audio');
        playBtn.disabled = false;
    }
    
    if (resetBtn) {
        resetBtn.classList.add('hidden');
    }

    if (currentlyPlayingBtn === playBtn) currentlyPlayingBtn = null;
    if (currentResetBtn === resetBtn) currentResetBtn = null;
}

function resetButton(btn) {
    btn.innerHTML = "<span>▶ Hear Captain Jim</span>";
    btn.classList.remove('playing-audio');
    btn.disabled = false;
}

askBtn.addEventListener('click', handleSearch);
userQuery.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSearch();
    }
});