import { Link } from 'react-router-dom';

export function NotFoundPage() {
  return (
    <div>
      <h1>404 - Page Not Found</h1>
      <Link to="/">Go back home</Link>
    </div>
  );
}

// export default NotFoundPage;