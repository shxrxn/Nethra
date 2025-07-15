import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../../core/themes/app_theme.dart';
import '../../../core/services/trust_service.dart';
import '../../../shared/widgets/behavioral_wrapper.dart';
import '../../../shared/widgets/trust_indicator.dart';
import '../../../shared/widgets/account_card.dart';
import '../../../shared/widgets/quick_actions.dart';
import '../../../shared/widgets/recent_transactions.dart';
import '../../trust_monitor/providers/trust_provider.dart';
import '../../trust_monitor/screens/trust_monitor_screen.dart';
import '../../transactions/screens/transactions_screen.dart';
import '../../mirage_interface/screens/mirage_screen.dart';
import '../../authentication/providers/auth_provider.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final trustProvider = Provider.of<TrustProvider>(context, listen: false);
      trustProvider.startMonitoring();
    });
  }

  @override
  Widget build(BuildContext context) {
    return BehavioralWrapper(
      child: Scaffold(
        backgroundColor: AppTheme.backgroundColor,
        body: SafeArea(
          child: Consumer<TrustProvider>(
            builder: (context, trustProvider, child) {
              // Show mirage interface if trust score is low
              if (trustProvider.shouldShowMirage) {
                return const MirageScreen();
              }
              
              return CustomScrollView(
                slivers: [
                  _buildAppBar(context),
                  SliverPadding(
                    padding: const EdgeInsets.all(16),
                    sliver: SliverList(
                      delegate: SliverChildListDelegate([
                        _buildTrustIndicator(trustProvider),
                        const SizedBox(height: 24),
                        _buildAccountCard().animate().slideX(delay: 300.ms),
                        const SizedBox(height: 24),
                        _buildQuickActions().animate().slideY(delay: 500.ms),
                        const SizedBox(height: 24),
                        _buildRecentTransactions().animate().fadeIn(delay: 700.ms),
                        const SizedBox(height: 24),
                        _buildSecurityInsights(trustProvider).animate().fadeIn(delay: 900.ms),
                      ]),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _buildAppBar(BuildContext context) {
    return SliverAppBar(
      expandedHeight: 100,
      floating: false,
      pinned: true,
      backgroundColor: AppTheme.backgroundColor,
      flexibleSpace: FlexibleSpaceBar(
        titlePadding: const EdgeInsets.only(left: 16, bottom: 16),
        title: Consumer<AuthProvider>(
          builder: (context, authProvider, child) {
            return Column(
              mainAxisAlignment: MainAxisAlignment.end,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Welcome back,',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: AppTheme.textSecondary,
                  ),
                ),
                Text(
                  authProvider.username ?? 'User',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            );
          },
        ),
      ),
      actions: [
        IconButton(
          icon: const Icon(Icons.notifications_outlined),
          onPressed: () {
            // Show notifications
          },
        ),
        IconButton(
          icon: const Icon(Icons.settings_outlined),
          onPressed: () {
            // Show settings
          },
        ),
        PopupMenuButton<String>(
          icon: const Icon(Icons.more_vert),
          onSelected: (value) {
            if (value == 'logout') {
              _handleLogout(context);
            }
          },
          itemBuilder: (context) => [
            const PopupMenuItem(
              value: 'logout',
              child: Row(
                children: [
                  Icon(Icons.logout),
                  SizedBox(width: 8),
                  Text('Logout'),
                ],
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildTrustIndicator(TrustProvider trustProvider) {
    return TrustIndicator(
      trustScore: trustProvider.trustScore,
      trustLevel: trustProvider.trustLevel,
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => const TrustMonitorScreen(),
          ),
        );
      },
    ).animate().slideY(delay: 100.ms);
  }

  Widget _buildAccountCard() {
    return const AccountCard();
  }

  Widget _buildQuickActions() {
    return QuickActions(
      onTransfer: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => const TransactionsScreen(),
          ),
        );
      },
      onPayBills: () {
        // Handle pay bills
      },
      onDeposit: () {
        // Handle deposit
      },
      onMoreActions: () {
        // Handle more actions
      },
    );
  }

  Widget _buildRecentTransactions() {
    return const RecentTransactions();
  }

  Widget _buildSecurityInsights(TrustProvider trustProvider) {
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
              const Icon(
                Icons.insights,
                color: AppTheme.primaryColor,
                size: 24,
              ),
              const SizedBox(width: 12),
              Text(
                'Security Insights',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildInsightItem(
            icon: Icons.check_circle,
            title: 'Behavioral Authentication',
            subtitle: 'Active and monitoring',
            color: AppTheme.successColor,
          ),
          const SizedBox(height: 12),
          _buildInsightItem(
            icon: Icons.security,
            title: 'Session Security',
            subtitle: 'Trust score: ${trustProvider.trustScore.toStringAsFixed(1)}',
            color: _getTrustColor(trustProvider.trustScore),
          ),
          const SizedBox(height: 12),
          _buildInsightItem(
            icon: Icons.device_hub,
            title: 'Device Recognition',
            subtitle: 'Trusted device',
            color: AppTheme.accentColor,
          ),
        ],
      ),
    );
  }

  Widget _buildInsightItem({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
  }) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(
            icon,
            color: color,
            size: 20,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                subtitle,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppTheme.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Color _getTrustColor(double trustScore) {
    if (trustScore >= 80) return AppTheme.successColor;
    if (trustScore >= 60) return AppTheme.accentColor;
    if (trustScore >= 40) return AppTheme.warningColor;
    return AppTheme.errorColor;
  }

  void _handleLogout(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Logout'),
        content: const Text('Are you sure you want to logout?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Provider.of<AuthProvider>(context, listen: false).logout();
              Provider.of<TrustProvider>(context, listen: false).stopMonitoring();
            },
            child: const Text('Logout'),
          ),
        ],
      ),
    );
  }
}