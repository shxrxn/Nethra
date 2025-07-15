import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/services/behavioral_service.dart';

class BehavioralWrapper extends StatefulWidget {
  final Widget child;
  
  const BehavioralWrapper({
    super.key,
    required this.child,
  });

  @override
  State<BehavioralWrapper> createState() => _BehavioralWrapperState();
}

class _BehavioralWrapperState extends State<BehavioralWrapper> {
  late final BehavioralService _behavioralService;
  
  @override
  void initState() {
    super.initState();
    _behavioralService = Provider.of<BehavioralService>(context, listen: false);
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        _recordTap(context);
      },
      onPanStart: (details) {
        _recordPanStart(details);
      },
      onPanUpdate: (details) {
        _recordPanUpdate(details);
      },
      onPanEnd: (details) {
        _recordPanEnd(details);
      },
      child: widget.child,
    );
  }
  
  void _recordTap(BuildContext context) {
    final RenderBox? box = context.findRenderObject() as RenderBox?;
    if (box != null) {
      final tapPosition = box.globalToLocal(Offset.zero);
      _behavioralService.recordTap(
        tapPosition.dx,
        tapPosition.dy,
        1.0, // Simulated pressure
        const Duration(milliseconds: 100), // Simulated duration
      );
    }
  }
  
  void _recordPanStart(DragStartDetails details) {
    // Record the start of a swipe gesture
  }
  
  void _recordPanUpdate(DragUpdateDetails details) {
    // Record swipe movement
  }
  
  void _recordPanEnd(DragEndDetails details) {
    // Record the end of a swipe gesture
    _behavioralService.recordSwipe(
      0.0, // Start position would be recorded in onPanStart
      0.0,
      details.localPosition.dx,
      details.localPosition.dy,
      details.velocity.pixelsPerSecond.dx,
      const Duration(milliseconds: 500), // Simulated duration
    );
  }
}