import "../CSS/styles.css";
import React from "react";
import { db } from "../firebase";
import styled from "styled-components";
import { useRecoileState, useSetRecoilState, useResetRecoilState } from "recoil";
import useState from "react";
import { async } from "@firebase/util";
import { collection, getDocs, addDoc } from "firebase/firestore";

function getpw() {
  const [users, setUsers] = useState([]);

  const usersCollectionRef = collection(db, "users");
  const getUsers = async () => {
    const usersCollection = await getDocs(usersCollectionRef);
    setUsers(usersCollection.docs.map((doc) => ({ ...doc.data(), id: doc.id })));
    console.log(usersCollection);
  };

  getUsers();
  const getpww = (uid) => {
    getUsers();
    console.log(users);
  };
}

export default getpw;
