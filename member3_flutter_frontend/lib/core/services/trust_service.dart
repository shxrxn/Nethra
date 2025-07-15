import 'dart:async';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'behavioral_service.dart';
import 'api_service.dart';
import '../../shared/models/trust_data.dart';
import '../../shared/models/behavioral_data.dart';

class TrustService {
  final BehavioralService _behavioralService;
  final ApiService _apiService;
  
  Timer? _trustUpdateTimer;
  double _currentTrustScore = 85.0;
  TrustLevel _currentTrustLevel = TrustLevel.high;
  
  TrustService(this._behavioralService, this._apiService);
  
  Stream<TrustData> get trustStream => _trustController.stream;
  final StreamController<TrustData> _trustController = StreamController<TrustData>.broadcast();
  
  void startTrustMonitoring() {
    _behavioralService.startMonitoring();
    
    _trustUpdateTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      _updateTrustScore();
    });
  }
  
  void stopTrustMonitoring() {
    _behavioralService.stopMonitoring();
    _trustUpdateTimer?.cancel();
    _trustController.close();
  }
  
  Future<void> _updateTrustScore() async {
    try {
      final behavioralData = _behavioralService.generateBehavioralData();
      final trustData = await _apiService.calculateTrustScore(behavioralData);
      
      _currentTrustScore = trustData.trustScore;
      _currentTrustLevel = trustData.trustLevel;
      
      _trustController.add(trustData);
    } catch (e) {
      if (kDebugMode) {
        print('Trust score update error: $e');
      }
      
      // Fallback to simulated trust score for demo
      _simulateTrustScore();
    }
  }
  
  void _simulateTrustScore() {
    // Simulate realistic trust score variations for demo
    final random = Random();
    final variation = (random.nextDouble() - 0.5) * 10;
    
    _currentTrustScore = (_currentTrustScore + variation).clamp(0.0, 100.0);
    _currentTrustLevel = _getTrustLevel(_currentTrustScore);
    
    final trustData = TrustData(
      trustScore: _currentTrustScore,
      trustLevel: _currentTrustLevel,
      riskFactors: _generateRiskFactors(),
      timestamp: DateTime.now(),
    );
    
    _trustController.add(trustData);
  }
  
  TrustLevel _getTrustLevel(double score) {
    if (score >= 80) return TrustLevel.high;
    if (score >= 60) return TrustLevel.medium;
    if (score >= 40) return TrustLevel.low;
    return TrustLevel.critical;
  }
  
  List<String> _generateRiskFactors() {
    final factors = <String>[];
    
    if (_currentTrustScore < 80) {
      factors.add('Unusual tap pattern detected');
    }
    if (_currentTrustScore < 60) {
      factors.add('Device orientation anomaly');
    }
    if (_currentTrustScore < 40) {
      factors.add('Suspicious navigation behavior');
    }
    if (_currentTrustScore < 20) {
      factors.add('Potential security threat');
    }
    
    return factors;
  }
  
  double get currentTrustScore => _currentTrustScore;
  TrustLevel get currentTrustLevel => _currentTrustLevel;
  
  bool get shouldActivateMirage => _currentTrustScore < 50;
  
  void recordSecurityEvent(String eventType, Map<String, dynamic> data) {
    // In a real implementation, this would send data to the backend
    if (kDebugMode) {
      print('Security event recorded: $eventType');
    }
  }
}