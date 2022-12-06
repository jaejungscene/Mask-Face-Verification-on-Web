import React from "react";
import styled from "styled-components";
import bg from "./image/bg.jpg";

const Container = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: url(${bg});
  background-size: cover;
`;
