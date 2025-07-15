import 'dart:math';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../../core/themes/app_theme.dart';
import '../../../shared/widgets/behavioral_wrapper.dart';
import '../providers/trust_provider.dart';
import '../../../shared/models/trust_data.dart';

class TrustMonitorScreen extends StatefulWidget {
  const TrustMonitorScreen({super.key});

  @override
  State<TrustMonitorScreen> createState() => _TrustMonitorScreenState();
}

class _TrustMonitorScreenState extends State<TrustMonitorScreen> {
  @override
  Widget build(BuildContext context) {
    return BehavioralWrapper(
      child: Scaffold(
        backgroundColor: AppTheme.backgroundColor,
        appBar: AppBar(
          title: const Text('Trust Monitor'),
          backgroundColor: AppTheme.backgroundColor,
          actions: [
            Consumer<TrustProvider>(
              builder: (context, trustProvider, child) {
                return PopupMenuButton<String>(
                  onSelected: (value) {
                    switch (value) {
                      case 'simulate_threat':
                        trustProvider.simulateThreat();
                        break;
                      case 'reset_trust':
                        trustProvider.resetTrust();
                        break;
                    }
                  },
                  itemBuilder: (context) => [
                    const PopupMenuItem(
                      value: 'simulate_threat',
                      child: Row(
                        children: [
                          Icon(Icons.warning, color: AppTheme.warningColor),
                          SizedBox(width: 8),
                          Text('Simulate Threat'),
                        ],
                      ),
                    ),
                    const PopupMenuItem(
                      value: 'reset_trust',
                      child: Row(
                        children: [
                          Icon(Icons.refresh, color: AppTheme.successColor),
                          SizedBox(width: 8),
                          Text('Reset Trust'),
                        ],
                      ),
                    ),
                  ],
                );
              },
            ),
          ],
        ),
        body: Consumer<TrustProvider>(
          builder: (context, trustProvider, child) {
            return SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildTrustScoreCard(trustProvider).animate().slideY(delay: 100.ms),
                  const SizedBox(height: 24),
                  _buildTrustChart(trustProvider).animate().slideX(delay: 300.ms),
                  const SizedBox(height: 24),
                  _buildRiskFactors(trustProvider).animate().fadeIn(delay: 500.ms),
                  const SizedBox(height: 24),
                  _buildBehavioralMetrics().animate().slideY(delay: 700.ms),
                  const SizedBox(height: 24),
                  _buildSecurityActions(trustProvider).animate().fadeIn(delay: 900.ms),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildTrustScoreCard(TrustProvider trustProvider) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            _getTrustColor(trustProvider.trustScore),
            _getTrustColor(trustProvider.trustScore).withOpacity(0.8),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: _getTrustColor(trustProvider.trustScore).withOpacity(0.3),
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
                  Text(
                    'Trust Score',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _getTrustLevelText(trustProvider.trustLevel),
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: Colors.white.withOpacity(0.9),
                    ),
                  ),
                ],
              ),
              Text(
                trustProvider.trustScore.toStringAsFixed(1),
                style: Theme.of(context).textTheme.displayLarge?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          LinearProgressIndicator(
            value: trustProvider.trustScore / 100,
            backgroundColor: Colors.white.withOpacity(0.3),
            valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
            minHeight: 8,
          ),
        ],
      ),
    );
  }

  Widget _buildTrustChart(TrustProvider trustProvider) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.surfaceColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Trust Score History',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 200,
            child: LineChart(
              LineChartData(
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  horizontalInterval: 20,
                  getDrawingHorizontalLine: (value) {
                    return FlLine(
                      color: Colors.grey.withOpacity(0.2),
                      strokeWidth: 1,
                    );
                  },
                ),
                titlesData: FlTitlesData(
                  show: true,
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 40,
                      getTitlesWidget: (value, meta) {
                        return Text(
                          value.toInt().toString(),
                          style: Theme.of(context).textTheme.bodySmall,
                        );
                      },
                    ),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 30,
                      getTitlesWidget: (value, meta) {
                        return Text(
                          '${value.toInt()}s',
                          style: Theme.of(context).textTheme.bodySmall,
                        );
                      },
                    ),
                  ),
                  rightTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  topTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                ),
                borderData: FlBorderData(show: false),
                lineBarsData: [
                  LineChartBarData(
                    spots: _generateTrustHistorySpots(trustProvider.trustScore),
                    isCurved: true,
                    color: _getTrustColor(trustProvider.trustScore),
                    barWidth: 3,
                    dotData: FlDotData(show: false),
                    belowBarData: BarAreaData(
                      show: true,
                      color: _getTrustColor(trustProvider.trustScore).withOpacity(0.1),
                    ),
                  ),
                ],
                minY: 0,
                maxY: 100,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRiskFactors(TrustProvider trustProvider) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.surfaceColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.warning_amber_rounded,
                color: trustProvider.riskFactors.isEmpty 
                    ? AppTheme.successColor 
                    : AppTheme.warningColor,
                size: 24,
              ),
              const SizedBox(width: 12),
              Text(
                'Risk Factors',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          if (trustProvider.riskFactors.isEmpty)
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.successColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.check_circle,
                    color: AppTheme.successColor,
                  ),
                  const SizedBox(width: 12),
                  Text(
                    'No risk factors detected',
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: AppTheme.successColor,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            )
          else
            ...trustProvider.riskFactors.map((factor) => Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.warningColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: AppTheme.warningColor.withOpacity(0.3),
                ),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.error_outline,
                    color: AppTheme.warningColor,
                    size: 20,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      factor,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppTheme.textPrimary,
                      ),
                    ),
                  ),
                ],
              ),
            )),
        ],
      ),
    );
  }

  Widget _buildBehavioralMetrics() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.surfaceColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Behavioral Metrics',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: _buildMetricCard(
                  title: 'Tap Pattern',
                  value: '95%',
                  icon: Icons.touch_app,
                  color: AppTheme.successColor,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildMetricCard(
                  title: 'Swipe Velocity',
                  value: '88%',
                  icon: Icons.swipe,
                  color: AppTheme.accentColor,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _buildMetricCard(
                  title: 'Device Tilt',
                  value: '92%',
                  icon: Icons.screen_rotation,
                  color: AppTheme.primaryColor,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildMetricCard(
                  title: 'Typing Rhythm',
                  value: '89%',
                  icon: Icons.keyboard,
                  color: AppTheme.warningColor,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetricCard({
    required String title,
    required String value,
    required IconData icon,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                icon,
                color: color,
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                value,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            title,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: AppTheme.textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSecurityActions(TrustProvider trustProvider) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.surfaceColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Security Actions',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () {
                    trustProvider.forceUpdateTrust();
                  },
                  icon: const Icon(Icons.refresh),
                  label: const Text('Update Trust'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryColor,
                    foregroundColor: Colors.white,
                  ),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () {
                    _showSecurityReport(context, trustProvider);
                  },
                  icon: const Icon(Icons.report),
                  label: const Text('Security Report'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.accentColor,
                    foregroundColor: Colors.white,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Color _getTrustColor(double trustScore) {
    if (trustScore >= 80) return AppTheme.successColor;
    if (trustScore >= 60) return AppTheme.accentColor;
    if (trustScore >= 40) return AppTheme.warningColor;
    return AppTheme.errorColor;
  }

  String _getTrustLevelText(TrustLevel trustLevel) {
    switch (trustLevel) {
      case TrustLevel.high:
        return 'High Trust';
      case TrustLevel.medium:
        return 'Medium Trust';
      case TrustLevel.low:
        return 'Low Trust';
      case TrustLevel.critical:
        return 'Critical Risk';
    }
  }

  List<FlSpot> _generateTrustHistorySpots(double currentScore) {
    final spots = <FlSpot>[];
    final random = Random();
    
    for (int i = 0; i < 20; i++) {
      final variation = (random.nextDouble() - 0.5) * 20;
      final score = (currentScore + variation).clamp(0.0, 100.0);
      spots.add(FlSpot(i.toDouble(), score));
    }
    
    return spots;
  }

  void _showSecurityReport(BuildContext context, TrustProvider trustProvider) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Security Report'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Trust Score: ${trustProvider.trustScore.toStringAsFixed(1)}'),
            Text('Trust Level: ${_getTrustLevelText(trustProvider.trustLevel)}'),
            const SizedBox(height: 16),
            const Text('Behavioral Patterns:'),
            const Text('• Tap patterns: Normal'),
            const Text('• Swipe velocity: Normal'),
            const Text('• Device handling: Normal'),
            const Text('• Navigation flow: Normal'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}