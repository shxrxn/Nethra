import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../core/themes/app_theme.dart';

class RecentTransactions extends StatelessWidget {
  const RecentTransactions({super.key});

  @override
  Widget build(BuildContext context) {
    final transactions = [
      {
        'title': 'Coffee Shop',
        'subtitle': 'Today, 2:30 PM',
        'amount': '-\$4.50',
        'icon': Icons.local_cafe,
        'color': AppTheme.warningColor,
      },
      {
        'title': 'Salary Deposit',
        'subtitle': 'Yesterday, 9:00 AM',
        'amount': '+\$3,200.00',
        'icon': Icons.work,
        'color': AppTheme.successColor,
      },
      {
        'title': 'Online Shopping',
        'subtitle': 'Dec 8, 2024',
        'amount': '-\$89.99',
        'icon': Icons.shopping_cart,
        'color': AppTheme.errorColor,
      },
      {
        'title': 'Transfer from John',
        'subtitle': 'Dec 7, 2024',
        'amount': '+\$250.00',
        'icon': Icons.person,
        'color': AppTheme.accentColor,
      },
    ];

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
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Recent Transactions',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              TextButton(
                onPressed: () {
                  // Navigate to full transaction history
                },
                child: Text(
                  'View All',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: AppTheme.primaryColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ...transactions.asMap().entries.map((entry) {
            final index = entry.key;
            final transaction = entry.value;
            
            return _buildTransactionItem(
              context,
              transaction: transaction,
            ).animate(delay: (index * 100).ms).slideX(begin: 0.3);
          }).toList(),
        ],
      ),
    );
  }

  Widget _buildTransactionItem(
    BuildContext context, {
    required Map<String, dynamic> transaction,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.backgroundColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.grey.withOpacity(0.1),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: (transaction['color'] as Color).withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              transaction['icon'] as IconData,
              color: transaction['color'] as Color,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  transaction['title'] as String,
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  transaction['subtitle'] as String,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: AppTheme.textSecondary,
                  ),
                ),
              ],
            ),
          ),
          Text(
            transaction['amount'] as String,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              fontWeight: FontWeight.bold,
              color: (transaction['amount'] as String).startsWith('+')
                  ? AppTheme.successColor
                  : AppTheme.errorColor,
            ),
          ),
        ],
      ),
    );
  }
}