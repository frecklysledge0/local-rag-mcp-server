# JWT Authentication Guide
This project uses standard JWT token-based authentication to secure all endpoints.
- Access Tokens: Valid for exactly 15 minutes.
- Refresh Tokens: Valid for exactly 7 days.
- Rotation: When a new access token is requested via `POST /api/auth/refresh`, a new refresh token is also rotated.
- Secure Storage: Refresh tokens must be stored in a secure, HttpOnly, SameSite=Strict cookie to prevent XSS attacks.
