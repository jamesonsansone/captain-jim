const askBtn = document.getElementById('askBtn');
const userQuery = document.getElementById('userQuery');
const loadingIndicator = document.getElementById('loadingIndicator');
const responseContainer = document.getElementById('responseContainer');
const summaryText = document.getElementById('summaryText');
const excerptsList = document.getElementById('excerptsList');

// -------------------------------------------------------------
// CONFIGURATION: PRODUCTION vs LOCAL
// -------------------------------------------------------------
const RENDER_URL = "https://captain-jim.onrender.com"; 
const LOCAL_URL = "http://127.0.0.1:8000";

// Force Render URL for consistency
const API_BASE_URL = RENDER_URL;
// -------------------------------------------------------------

let currentAudio = new Audio();
let currentlyPlayingBtn = null; // To track active button
let currentResetBtn = null;     // To track active reset button

// Just fills the text, does NOT fire search automatically
function fillQuery(text) {
    userQuery.value = text;
}

async function handleSearch() {
    const query = userQuery.value.trim();
    if (!query) return;

    loadingIndicator.classList.remove('hidden');
    responseContainer.classList.add('hidden');
    stopAudio(); // Stop any playing audio on new search

    try {
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query })
        });

        const data = await response.json();

        if (response.ok) {
            // 1. Set the Historian Summary (Text Only)
            summaryText.innerText = data.summary;

            // 2. Build Excerpts with Dual-Button Interface (Play | Reset)
            excerptsList.innerHTML = '';
            data.excerpts.forEach((excerpt, index) => {
                
                // Format the source text (Remove .txt, underscores)
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
            alert("Error retrieving archives.");
            loadingIndicator.classList.add('hidden');
        }

    } catch (error) {
        console.error(error);
        alert("Server error. Ensure server is running.");
        loadingIndicator.classList.add('hidden');
    }
}

function escapeHtml(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

async function playExcerptAudio(btnElement, textToSpeak, index) {
    const relatedResetBtn = document.getElementById(`reset-btn-${index}`);

    // LOGIC: If currently playing THIS button, Toggle Pause/Play
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

    // LOGIC: New Button Clicked -> Stop previous
    stopAudio();

    // --- MOBILE FIX: PRIME THE AUDIO PUMP ---
    // We play silence immediately to "unlock" the mobile audio channel
    // while we wait for the slow server response.
    currentAudio = new Audio(); // Create fresh instance
    currentAudio.play().catch(() => {}); // Play nothing; ignore empty error
    // ----------------------------------------

    // Visual Loading State
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
            const url = URL.createObjectURL(blob);
            
            // Swap the silent source for the real source
            currentAudio.src = url;
            currentAudio.play(); // This will now work on mobile!
            
            // Set Active States
            currentlyPlayingBtn = btnElement;
            currentResetBtn = relatedResetBtn;

            // Update UI: Show Pause, Show Reset
            btnElement.classList.add('playing-audio'); 
            btnElement.innerHTML = "<span>II Pause</span>";
            btnElement.disabled = false;
            
            if (relatedResetBtn) relatedResetBtn.classList.remove('hidden');

            // When audio finishes naturally
            currentAudio.onended = () => {
                resetSpecificAudio(index);
            };

        } else {
            alert("Audio Error");
            resetSpecificAudio(index);
        }
    } catch (e) {
        console.error(e);
        alert("Connection Error");
        resetSpecificAudio(index);
    }
}

// Global Stop (used when switching tracks or searching)
function stopAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    // Reset UI of whatever was playing
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

// Specific Reset for the little buttons
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

    // Clear global trackers if they matched this specific index
    if (currentlyPlayingBtn === playBtn) currentlyPlayingBtn = null;
    if (currentResetBtn === resetBtn) currentResetBtn = null;
}

askBtn.addEventListener('click', handleSearch);
userQuery.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSearch();
    }
});