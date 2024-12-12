import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
} from "react-router-dom";
import "./style.css";
import Home from "./pages/Home"; // Votre code principal
import SecondPage from "./pages/ChatHistory"; // La deuxième page

function App() {
  return (
    <Router>
      <nav className="navbar">
        <div className="navbar-logo">
          <NavLink to="/">
            <img src="./logo192.png" alt="Logo" className="logo-image" />
          </NavLink>
        </div>
        <ul className="navbar-list">
          <li className="navbar-item">
            <NavLink
              to="/"
              className={({ isActive }) =>
                isActive ? "navbar-link active" : "navbar-link"
              }
            >
              Home
            </NavLink>
          </li>
          <li className="navbar-item">
            <NavLink
              to="/chat_history"
              className={({ isActive }) =>
                isActive ? "navbar-link active" : "navbar-link"
              }
            >
              Chat History
            </NavLink>
          </li>
        </ul>
      </nav>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat_history" element={<SecondPage />} />
      </Routes>
    </Router>
  );
}

export default App;
