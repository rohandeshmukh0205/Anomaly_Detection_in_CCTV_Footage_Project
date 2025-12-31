// static/firebase-messaging-sw.js
importScripts('https://www.gstatic.com/firebasejs/10.12.5/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.5/firebase-messaging-compat.js');

firebase.initializeApp({
  apiKey: "AIzaSyBP2qiP9xOq2n-ueg48egQHK9qlteUMP5o",
  authDomain: "anomaly-detection-in-cctv.firebaseapp.com",
  projectId: "anomaly-detection-in-cctv",
  storageBucket: "anomaly-detection-in-cctv.firebasestorage.app",
  messagingSenderId: "1016994415144",
  appId: "1:1016994415144:web:070ba602553c0c3c9f6b1d",
});

firebase.messaging();
