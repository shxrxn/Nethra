import 'dart:convert';
import 'package:http/http.dart' as http;
import '../constants/app_constants.dart';
import '../../shared/models/trust_data.dart';
import '../../shared/models/behavioral_data.dart';

class ApiService {
  static const String baseUrl = AppConstants.baseUrl;
  
  Future<TrustData> calculateTrustScore(BehavioralData behavioralData) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/trust/calculate'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(behavioralData.toJson()),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return TrustData.fromJson(data);
      } else {
        throw Exception('Failed to calculate trust score');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> validateTransaction(Map<String, dynamic> transactionData) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/transaction/validate'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(transactionData),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to validate transaction');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> reportTamperAttempt(Map<String, dynamic> tamperData) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/security/tamper-report'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(tamperData),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to report tamper attempt');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<bool> authenticateUser(String username, String password) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/auth/login'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'username': username,
          'password': password,
        }),
      );

      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}