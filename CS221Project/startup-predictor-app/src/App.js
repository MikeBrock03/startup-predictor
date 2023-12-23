import logo from './logo.svg';
import './App.css';
import StartupForm from './StartupForm';
import { Analytics } from '@vercel/analytics/react';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Judge My Startup</h1>
      </header>
      <main>
        <StartupForm/>
        <Analytics/>
      </main>
    </div>
  );
}

export default App;
