enum TrustLevel {
  critical,
  low,
  medium,
  high,
}

class TrustData {
  final double trustScore;
  final TrustLevel trustLevel;
  final List<String> riskFactors;
  final DateTime timestamp;
  
  TrustData({
    required this.trustScore,
    required this.trustLevel,
    required this.riskFactors,
    required this.timestamp,
  });
  
  Map<String, dynamic> toJson() {
    return {
      'trust_score': trustScore,
      'trust_level': trustLevel.name,
      'risk_factors': riskFactors,
      'timestamp': timestamp.toIso8601String(),
    };
  }
  
  factory TrustData.fromJson(Map<String, dynamic> json) {
    return TrustData(
      trustScore: json['trust_score'].toDouble(),
      trustLevel: TrustLevel.values.firstWhere(
        (e) => e.name == json['trust_level'],
        orElse: () => TrustLevel.medium,
      ),
      riskFactors: List<String>.from(json['risk_factors']),
      timestamp: DateTime.parse(json['timestamp']),
    );
  }
}