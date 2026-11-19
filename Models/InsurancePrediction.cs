using Microsoft.ML.Data;

namespace InsuranceRegression.Models;

public class InsurancePrediction
{
    [ColumnName("Score")]
    public float PredictedCharges { get; set; }
}
