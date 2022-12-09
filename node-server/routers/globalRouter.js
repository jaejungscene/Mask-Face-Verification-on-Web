import express from "express";
import login from "../controllers/homeController";

const globalRouter = express.Router();

globalRouter.get("/", login);

export default globalRouter;
