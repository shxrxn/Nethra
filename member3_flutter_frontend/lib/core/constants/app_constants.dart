class AppConstants {
  static const String appName = 'NETHRA Banking';
  static const String baseUrl = 'http://10.0.2.2:8000'; // Android emulator
  // For iOS simulator, use: 'http://127.0.0.1:8000'
  // For physical device, use your computer's IP: 'http://192.168.1.XXX:8000'
  
  // Trust Score Thresholds
  static const double highTrustThreshold = 80.0;
  static const double mediumTrustThreshold = 60.0;
  static const double lowTrustThreshold = 40.0;
  static const double mirageActivationThreshold = 50.0;
  
  // Security Settings
  static const int maxFailedAttempts = 3;
  static const int sessionTimeoutMinutes = 30;
  static const int trustUpdateIntervalSeconds = 5;
  
  // Demo Data
  static const String demoUsername = 'demo_user';
  static const String demoPassword = 'demo123';
  static const double demoAccountBalance = 25750.50;
  
  // App Colors
  static const int primaryColorValue = 0xFF1E88E5;
  static const int successColorValue = 0xFF4CAF50;
  static const int warningColorValue = 0xFFFF9800;
  static const int errorColorValue = 0xFFE53935;
}