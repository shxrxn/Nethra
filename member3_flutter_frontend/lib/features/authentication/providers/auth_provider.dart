import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../../core/services/api_service.dart';
import '../../../core/constants/app_constants.dart';

class AuthProvider with ChangeNotifier {
  bool _isAuthenticated = false;
  bool _isLoading = false;
  String? _errorMessage;
  String? _userId;
  String? _username;
  
  bool get isAuthenticated => _isAuthenticated;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  String? get userId => _userId;
  String? get username => _username;
  
  Future<void> checkAuthStatus() async {
    final prefs = await SharedPreferences.getInstance();
    _isAuthenticated = prefs.getBool('isAuthenticated') ?? false;
    _userId = prefs.getString('userId');
    _username = prefs.getString('username');
    notifyListeners();
  }
  
  Future<bool> login(String username, String password) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();
    
    try {
      // Demo authentication
      if (username == AppConstants.demoUsername && password == AppConstants.demoPassword) {
        await _setAuthenticatedState(true, 'demo_user_id', username);
        return true;
      }
      
      // API authentication
      final apiService = ApiService();
      final success = await apiService.authenticateUser(username, password);
      
      if (success) {
        await _setAuthenticatedState(true, 'user_id', username);
        return true;
      } else {
        _errorMessage = 'Invalid username or password';
        return false;
      }
    } catch (e) {
      _errorMessage = 'Authentication failed. Please try again.';
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
  
  Future<void> logout() async {
    await _setAuthenticatedState(false, null, null);
  }
  
  Future<void> _setAuthenticatedState(bool isAuthenticated, String? userId, String? username) async {
    final prefs = await SharedPreferences.getInstance();
    
    _isAuthenticated = isAuthenticated;
    _userId = userId;
    _username = username;
    
    await prefs.setBool('isAuthenticated', isAuthenticated);
    if (userId != null) {
      await prefs.setString('userId', userId);
    } else {
      await prefs.remove('userId');
    }
    
    if (username != null) {
      await prefs.setString('username', username);
    } else {
      await prefs.remove('username');
    }
    
    notifyListeners();
  }
  
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }
}