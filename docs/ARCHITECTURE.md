# System Architecture & Technical Specifications

## Architectural Layers

SmartAgri AI is built using **Clean Architecture** combined with the **MVVM (Model-View-ViewModel)** design pattern.

```
+-------------------------------------------------------------+
|                      UI Layer (Compose)                     |
|   DashboardScreen, ScannerScreen, AssistantScreen, etc.     |
+------------------------------+------------------------------+
                               |
+------------------------------v------------------------------+
|                    ViewModel Layer (StateFlow)              |
|   DashboardViewModel, ScannerViewModel, WeatherViewModel    |
+------------------------------+------------------------------+
                               |
+------------------------------v------------------------------+
|                     Domain Layer (Use Cases)                |
|  AnalyzeCropHealth, CalculateScore, GetIrrigationAdvice     |
+------------------------------+------------------------------+
                               |
+------------------------------v------------------------------+
|                 Repository Interfaces & Data Layer          |
|  CropRepositoryImpl, AIRepositoryImpl, WeatherRepositoryImpl|
+-------------+----------------+----------------+-------------+
              |                |                |
+-------------v---+   +--------v-------+   +----+------------+
| Room Database   |   | TFLite Engine  |   | Retrofit API    |
| (SQLite Local)  |   | (On-Device ML) |   | (Weather/LLM)   |
+-----------------+   +----------------+   +-----------------+
```

## ML Inference Pipeline Architecture
1. **Input Normalization**: CameraX Bitmap converted to $224 \times 224 \times 3$ FloatBuffer.
2. **Interpreter Runner**: `TFLiteCropClassifier` allocates hardware accelerators (NNAPI/GPU Delegate) if available, falling back to CPU.
3. **Agronomic Mapping**: Confidence score computed via Softmax; mapped to target crop species and health condition.

## Database Schema (Room)
- `crops`: `id`, `name`, `variety`, `plantingDate`, `fieldName`, `fieldSize`, `location`, `notes`, `createdAt`.
- `scans`: `id`, `date`, `cropName`, `healthScore`, `issueDetected`, `confidence`, `explanation`, `recommendations`, `imagePath`.
- `chat_messages`: `id`, `sender` (USER/AI), `text`, `timestamp`.
