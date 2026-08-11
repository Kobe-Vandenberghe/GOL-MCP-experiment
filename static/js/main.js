"use strict";

// Entry point. Importing controls.js is what wires up the input handlers.

import { draw } from "./renderer.js";
import { connect } from "./socket.js";
import "./controls.js";

connect();
draw();
