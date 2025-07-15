import 'package:flutter/foundation.dart';
import 'dart:async';
import '../../../core/services/trust_service.dart';
import '../../../core/services/behavioral_service.dart';
import '../../../core/services/api_service.dart';
import '../../../shared/models/trust_data.dart';

class TrustProvider with ChangeNotifier {
  late final TrustService _trustService;
  StreamSubscription<TrustData>? _trustSubscription;
  
  double _trustScore = 85.0;
  TrustLevel _trustLevel = TrustLevel.high;
  List<String> _riskFactors = [];
  bool _isMonitoring = false;
  bool _shouldShowMirage = false;
  
  TrustProvider() {
    final behavioralService = BehavioralService();
    final apiService = ApiService();
    _trustService = TrustService(behavioralService, apiService);
  }
  
  double get trustScore => _trustScore;
  TrustLevel get trustLevel => _trustLevel;
  List<String> get riskFactors => _riskFactors;
  bool get isMonitoring => _isMonitoring;
  bool get shouldShowMirage => _shouldShowMirage;
  
  void startMonitoring() {
    if (_isMonitoring) return;
    
    _isMonitoring = true;
    _trustService.startTrustMonitoring();
    
    _trustSubscription = _trustService.trustStream.listen((trustData) {
      _trustScore = trustData.trustScore;
      _trustLevel = trustData.trustLevel;
      _riskFactors = trustData.riskFactors;
      _shouldShowMirage = _trustScore < 50;
      
      notifyListeners();
      
      // Log security events
      if (_trustScore < 60) {
        _trustService.recordSecurityEvent('LOW_TRUST_SCORE', {
          'score': _trustScore,
          'level': _trustLevel.name,
          'factors': _riskFactors,
        });
      }
    });
    
    notifyListeners();
  }
  
  void stopMonitoring() {
    if (!_isMonitoring) return;
    
    _isMonitoring = false;
    _trustService.stopTrustMonitoring();
    _trustSubscription?.cancel();
    _trustSubscription = null;
    
    notifyListeners();
  }
  
  void forceUpdateTrust() {
    // Force an immediate trust score update
    _trustService.recordSecurityEvent('MANUAL_UPDATE', {
      'timestamp': DateTime.now().toIso8601String(),
    });
  }
  
  void simulateThreat() {
    // Simulate a security threat for demo purposes
    _trustScore = 25.0;
    _trustLevel = TrustLevel.critical;
    _riskFactors = [
      'Suspicious tap patterns detected',
      'Unusual device movement',
      'Potential unauthorized access',
    ];
    _shouldShowMirage = true;
    
    notifyListeners();
    
    _trustService.recordSecurityEvent('SIMULATED_THREAT', {
      'score': _trustScore,
      'level': _trustLevel.name,
      'factors': _riskFactors,
    });
  }
  
  void resetTrust() {
    // Reset trust score to normal for demo purposes
    _trustScore = 85.0;
    _trustLevel = TrustLevel.high;
    _riskFactors = [];
    _shouldShowMirage = false;
    
    notifyListeners();
    
    _trustService.recordSecurityEvent('TRUST_RESET', {
      'score': _trustScore,
      'level': _trustLevel.name,
    });
  }
  
  @override
  void dispose() {
    stopMonitoring();
    super.dispose();
  }
}