// ========== AI Fire Detection - Webcam Script ==========
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("webcamStatus");

let streaming = false;
let intervalId = null;
let sending = false;
let fireActive = false; // Track if fire is currently detected

// ===== Start Camera =====
document.getElementById("start").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    statusText.textContent = "Status: camera started ✅";
    streaming = true;

    const interval = parseInt(document.getElementById("interval").value) || 1000;
    sendFrames(interval);
  } catch (err) {
    console.error("Camera access denied:", err);
    alert("⚠️ Unable to access camera. Please allow camera permissions.");
  }
});

// ===== Stop Camera =====
document.getElementById("stop").addEventListener("click", () => {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
  }
  clearInterval(intervalId);
  streaming = false;
  statusText.textContent = "Status: camera stopped ⛔";
});

// ===== Capture & Send Frame =====
function sendFrames(interval) {
  intervalId = setInterval(async () => {
    if (!streaming || sending) return;
    sending = true;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL("image/jpeg");

    try {
      const res = await fetch("/predict_webcam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      const data = await res.json();
      if (data.fire === true) {
        if (!fireActive) {
          fireActive = true;
          playAlarm();
          alert("🔥 Fire Detected in Live Feed!");
        }
      } else {
        fireActive = false;
        stopAlarm();
      }
    } catch (err) {
      console.error("Error sending frame:", err);
    } finally {
      sending = false;
    }
  }, interval);
}

// ===== Play/Stop Alarm =====

function playAlarm() {
  const audio = document.getElementById("alarmAudio");
  if (audio.paused) {
    audio.play();
  }
}

function stopAlarm() {
  const audio = document.getElementById("alarmAudio");
  audio.pause();
  audio.currentTime = 0;
}

document.getElementById("stopAlarm").addEventListener("click", () => {
  stopAlarm();
  fireActive = false;
});


