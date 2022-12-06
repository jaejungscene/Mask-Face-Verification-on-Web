import { atom } from "recoil";

export const usersAtom = atom({
  key: "usersAtom",
  default: [],
});

export default usersAtom;
