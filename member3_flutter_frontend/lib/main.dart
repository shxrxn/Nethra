import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'core/themes/app_theme.dart';
import 'core/services/behavioral_service.dart';
import 'core/services/api_service.dart';
import 'core/services/trust_service.dart';
import 'features/authentication/screens/login_screen.dart';
import 'features/dashboard/screens/dashboard_screen.dart';
import 'features/trust_monitor/providers/trust_provider.dart';
import 'features/authentication/providers/auth_provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Set system UI overlay style
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      systemNavigationBarColor: Colors.white,
      systemNavigationBarIconBrightness: Brightness.dark,
    ),
  );

  runApp(const NethraBankingApp());
}

class NethraBankingApp extends StatelessWidget {
  const NethraBankingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => TrustProvider()),
        Provider(create: (_) => ApiService()),
        Provider(create: (_) => BehavioralService()),
        ProxyProvider2<BehavioralService, ApiService, TrustService>(
          create: (context) => TrustService(
            Provider.of<BehavioralService>(context, listen: false),
            Provider.of<ApiService>(context, listen: false),
          ),
          update: (context, behavioralService, apiService, previous) => 
            TrustService(behavioralService, apiService),
        ),
      ],
      child: MaterialApp(
        title: 'NETHRA Banking',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.lightTheme,
        home: Consumer<AuthProvider>(
          builder: (context, authProvider, child) {
            return authProvider.isAuthenticated
                ? const DashboardScreen()
                : const LoginScreen();
          },
        ),
      ),
    );
  }
}