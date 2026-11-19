using InsuranceRegression.Evaluation;
using InsuranceRegression.Models;

namespace InsuranceRegression.Reporting;

// Виведення результатів у консоль
public static class ConsoleReporter
{
    public static void PrintComparisonHeader()
    {
        Console.WriteLine("Порівняння регресійних моделей на варіанті 5 (прогноз медичного страхування)");
        Console.WriteLine(new string('-', 75));
    }

    public static void PrintComparisonRow(TrainerResult result)
    {
        Console.WriteLine($"{result.Name,-28} R^2 = {result.Metrics.RSquared:F4}   MAE = {result.Metrics.MeanAbsoluteError,10:F2}   RMSE = {result.Metrics.RootMeanSquaredError,10:F2}");
    }

    public static void PrintBest(TrainerResult best)
    {
        Console.WriteLine(new string('-', 75));
        Console.WriteLine($"Найкраща модель: {best.Name} (R^2 = {best.RSquared:F4})");
    }

    public static void PrintSamplePrediction(InsurancePrediction result)
    {
        Console.WriteLine();
        Console.WriteLine("Приклад прогнозу для нового клієнта:");
        Console.WriteLine("  Вік=35, стать=male, ІМТ=28.5, дітей=2, курець=yes, регіон=southeast");
        Console.WriteLine($"  Прогнозовані витрати на страхування: {result.PredictedCharges:F2} $");
    }
}
