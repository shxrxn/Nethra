import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../core/themes/app_theme.dart';
import '../models/trust_data.dart';

class TrustIndicator extends StatelessWidget {
  final double trustScore;
  final TrustLevel trustLevel;
  final VoidCallback? onTap;

  const TrustIndicator({
    super.key,
    required this.trustScore,
    required this.trustLevel,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              _getTrustColor(trustLevel),
              _getTrustColor(trustLevel).withOpacity(0.8),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: _getTrustColor(trustLevel).withOpacity(0.3),
              blurRadius: 20,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _getTrustIcon(trustLevel),
                          color: Colors.white,
                          size: 24,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Trust Score',
                          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _getTrustLevelText(trustLevel),
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: Colors.white.withOpacity(0.9),
                      ),
                    ),
                  ],
                ),
                Text(
                  trustScore.toStringAsFixed(1),
                  style: Theme.of(context).textTheme.displayLarge?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            Stack(
              children: [
                Container(
                  height: 8,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.3),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
                AnimatedContainer(
                  duration: const Duration(milliseconds: 1000),
                  height: 8,
                  width: MediaQuery.of(context).size.width * 0.8 * (trustScore / 100),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Behavioral Authentication Active',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.white.withOpacity(0.9),
                  ),
                ),
                Icon(
                  Icons.arrow_forward_ios,
                  color: Colors.white.withOpacity(0.7),
                  size: 16,
                ),
              ],
            ),
          ],
        ),
      ),
    ).animate().slideY(begin: 0.3, duration: 600.ms).fadeIn();
  }

  Color _getTrustColor(TrustLevel trustLevel) {
    switch (trustLevel) {
      case TrustLevel.high:
        return AppTheme.successColor;
      case TrustLevel.medium:
        return AppTheme.accentColor;
      case TrustLevel.low:
        return AppTheme.warningColor;
      case TrustLevel.critical:
        return AppTheme.errorColor;
    }
  }

  IconData _getTrustIcon(TrustLevel trustLevel) {
    switch (trustLevel) {
      case TrustLevel.high:
        return Icons.verified_user;
      case TrustLevel.medium:
        return Icons.security;
      case TrustLevel.low:
        return Icons.warning;
      case TrustLevel.critical:
        return Icons.error;
    }
  }

  String _getTrustLevelText(TrustLevel trustLevel) {
    switch (trustLevel) {
      case TrustLevel.high:
        return 'High Trust - Secure';
      case TrustLevel.medium:
        return 'Medium Trust - Monitoring';
      case TrustLevel.low:
        return 'Low Trust - Caution';
      case TrustLevel.critical:
        return 'Critical Risk - Alert';
    }
  }
}