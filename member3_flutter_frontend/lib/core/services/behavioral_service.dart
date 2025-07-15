import 'dart:async';
import 'dart:math';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:flutter/services.dart';
import '../../shared/models/behavioral_data.dart';

class BehavioralService {
  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroscopeSubscription;
  
  final List<AccelerometerEvent> _accelerometerData = [];
  final List<GyroscopeEvent> _gyroscopeData = [];
  final List<TapData> _tapData = [];
  final List<SwipeData> _swipeData = [];
  
  DateTime? _sessionStartTime;
  int _tapCount = 0;
  double _totalSwipeDistance = 0.0;
  
  void startMonitoring() {
    _sessionStartTime = DateTime.now();
    
    _accelerometerSubscription = accelerometerEvents.listen((event) {
      _accelerometerData.add(event);
      if (_accelerometerData.length > 100) {
        _accelerometerData.removeAt(0);
      }
    });
    
    _gyroscopeSubscription = gyroscopeEvents.listen((event) {
      _gyroscopeData.add(event);
      if (_gyroscopeData.length > 100) {
        _gyroscopeData.removeAt(0);
      }
    });
  }
  
  void stopMonitoring() {
    _accelerometerSubscription?.cancel();
    _gyroscopeSubscription?.cancel();
  }
  
  void recordTap(double x, double y, double pressure, Duration duration) {
    _tapCount++;
    _tapData.add(TapData(
      x: x,
      y: y,
      pressure: pressure,
      duration: duration,
      timestamp: DateTime.now(),
    ));
    
    if (_tapData.length > 50) {
      _tapData.removeAt(0);
    }
  }
  
  void recordSwipe(double startX, double startY, double endX, double endY, 
                   double velocity, Duration duration) {
    final distance = sqrt(pow(endX - startX, 2) + pow(endY - startY, 2));
    _totalSwipeDistance += distance;
    
    _swipeData.add(SwipeData(
      startX: startX,
      startY: startY,
      endX: endX,
      endY: endY,
      velocity: velocity,
      duration: duration,
      timestamp: DateTime.now(),
    ));
    
    if (_swipeData.length > 50) {
      _swipeData.removeAt(0);
    }
  }
  
  BehavioralData generateBehavioralData() {
    final currentTime = DateTime.now();
    final sessionDuration = _sessionStartTime != null
        ? currentTime.difference(_sessionStartTime!).inSeconds
        : 0;
    
    return BehavioralData(
      sessionDuration: sessionDuration,
      tapCount: _tapCount,
      averageTapPressure: _calculateAverageTapPressure(),
      averageTapDuration: _calculateAverageTapDuration(),
      swipeCount: _swipeData.length,
      averageSwipeVelocity: _calculateAverageSwipeVelocity(),
      totalSwipeDistance: _totalSwipeDistance,
      deviceTiltVariation: _calculateDeviceTiltVariation(),
      movementPattern: _calculateMovementPattern(),
      typingRhythm: _calculateTypingRhythm(),
      navigationFlow: _calculateNavigationFlow(),
      timestamp: currentTime,
    );
  }
  
  double _calculateAverageTapPressure() {
    if (_tapData.isEmpty) return 0.0;
    return _tapData.map((e) => e.pressure).reduce((a, b) => a + b) / _tapData.length;
  }
  
  double _calculateAverageTapDuration() {
    if (_tapData.isEmpty) return 0.0;
    return _tapData.map((e) => e.duration.inMilliseconds).reduce((a, b) => a + b) / _tapData.length;
  }
  
  double _calculateAverageSwipeVelocity() {
    if (_swipeData.isEmpty) return 0.0;
    return _swipeData.map((e) => e.velocity).reduce((a, b) => a + b) / _swipeData.length;
  }
  
  double _calculateDeviceTiltVariation() {
    if (_accelerometerData.isEmpty) return 0.0;
    
    final tilts = _accelerometerData.map((e) => 
      sqrt(e.x * e.x + e.y * e.y + e.z * e.z));
    
    if (tilts.isEmpty) return 0.0;
    
    final average = tilts.reduce((a, b) => a + b) / tilts.length;
    final variance = tilts.map((e) => pow(e - average, 2)).reduce((a, b) => a + b) / tilts.length;
    
    return sqrt(variance);
  }
  
  List<double> _calculateMovementPattern() {
    if (_accelerometerData.isEmpty) return [0.0, 0.0, 0.0];
    
    final xAvg = _accelerometerData.map((e) => e.x).reduce((a, b) => a + b) / _accelerometerData.length;
    final yAvg = _accelerometerData.map((e) => e.y).reduce((a, b) => a + b) / _accelerometerData.length;
    final zAvg = _accelerometerData.map((e) => e.z).reduce((a, b) => a + b) / _accelerometerData.length;
    
    return [xAvg, yAvg, zAvg];
  }
  
  double _calculateTypingRhythm() {
    if (_tapData.length < 2) return 0.0;
    
    final intervals = <double>[];
    for (int i = 1; i < _tapData.length; i++) {
      final interval = _tapData[i].timestamp.difference(_tapData[i-1].timestamp).inMilliseconds;
      intervals.add(interval.toDouble());
    }
    
    if (intervals.isEmpty) return 0.0;
    
    final average = intervals.reduce((a, b) => a + b) / intervals.length;
    return average;
  }
  
  double _calculateNavigationFlow() {
    // This would be enhanced with actual navigation tracking
    return Random().nextDouble() * 100;
  }
}

class TapData {
  final double x;
  final double y;
  final double pressure;
  final Duration duration;
  final DateTime timestamp;
  
  TapData({
    required this.x,
    required this.y,
    required this.pressure,
    required this.duration,
    required this.timestamp,
  });
}

class SwipeData {
  final double startX;
  final double startY;
  final double endX;
  final double endY;
  final double velocity;
  final Duration duration;
  final DateTime timestamp;
  
  SwipeData({
    required this.startX,
    required this.startY,
    required this.endX,
    required this.endY,
    required this.velocity,
    required this.duration,
    required this.timestamp,
  });
}