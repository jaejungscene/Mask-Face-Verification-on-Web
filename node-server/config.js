import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyDS9PgYr95C4HKoPtPdgJwD5EFUXDtn9ug",
  authDomain: "it2-teamproject.firebaseapp.com",
  projectId: "it2-teamproject",
  storageBucket: "it2-teamproject.appspot.com",
  messagingSenderId: "984166777628",
  appId: "1:984166777628:web:ce5701032c8c889094b0d9",
  measurementId: "G-TF1QY8H7Z3",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// const analytics = firebase.getAnalytics(app);

export const firebaseAuth = getAuth(app);
export const db = getFirestore(app);
// export const User = db.collection("users");
// export const fStorage = getStorage(app);
