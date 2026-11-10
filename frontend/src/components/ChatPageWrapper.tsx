import {ChatPage} from './ChatPage';
import { useNavigate, useParams } from 'react-router-dom';

export function ChatPageWrapper() {
    const { sessionId } = useParams();
    const navigate = useNavigate();
    console.log('sessionId:', sessionId);
    console.log('Type of sessionId:', typeof sessionId ); // Debugging line
 if(!sessionId||sessionId.trim() === '') {
            navigate('/404');
        }
    else{

        return (
            //    <h1>Chat Page for Session: {sessionId}</h1>
            <ChatPage sessionId={sessionId } />
        );
    }
}