
import { HomePage } from './components/HomePage';
import { ChatPageWrapper } from './components/ChatPageWrapper';
import { NotFoundPage } from './components/NotFoundPage';

import {BrowserRouter as Router, Route, Routes} from 'react-router-dom'
function App() {
  
  return (
    <div style={{ fontFamily: "system-ui, sans-serif" }}>
     <Router>
      <Routes>
        <Route  path='/' element={<HomePage />} />
        <Route  path='/chat/:sessionId' element={<ChatPageWrapper/>} />
        <Route  path='*' element={<NotFoundPage />} />
        
      </Routes>
     </Router>
    </div>
  );
}

export default App;
