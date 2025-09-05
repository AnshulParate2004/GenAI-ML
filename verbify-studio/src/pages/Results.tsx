import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { BarChart3, CheckCircle, Lightbulb, Target, BookOpen, Presentation } from "lucide-react";

const Results = () => {
  const navigate = useNavigate();

  // Mock data for demonstration
  const overallScore = 85;
  const categories = [
    { name: "Communication Skills", score: 90, description: "Excellent verbal communication and clarity" },
    { name: "Technical Knowledge", score: 82, description: "Strong understanding of technical concepts" },
    { name: "Problem Solving", score: 88, description: "Great analytical thinking and approach" },
    { name: "Confidence", score: 79, description: "Good presence, room for improvement" },
  ];

  const feedback = [
    {
      type: "strength",
      title: "Strong Communication",
      description: "You articulated your thoughts clearly and maintained good eye contact throughout the interview.",
    },
    {
      type: "improvement",
      title: "Technical Depth",
      description: "Consider providing more detailed examples when discussing technical implementations.",
    },
    {
      type: "strength", 
      title: "Problem-Solving Approach",
      description: "Your systematic approach to breaking down complex problems was impressive.",
    },
  ];

  return (
    <div className="min-h-screen pt-16 bg-background">
      <div className="container mx-auto px-6 py-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold mb-8 text-center bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">
            Interview Results
          </h1>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Overall Score Card */}
            <div className="lg:col-span-1">
              <Card className="p-8 text-center bg-card/50 backdrop-blur-sm border-border/50 shadow-card">
                <BarChart3 className="w-16 h-16 text-primary mx-auto mb-4" />
                <h2 className="text-2xl font-semibold mb-4">Overall Score</h2>
                <div className="text-6xl font-bold text-primary mb-4">{overallScore}%</div>
                <div className="text-muted-foreground mb-6">
                  Excellent performance! You're in the top 15% of candidates.
                </div>
                <Button
                  onClick={() => navigate("/start-interview")}
                  className="bg-gradient-primary hover:opacity-90 text-primary-foreground px-6 py-3 rounded-xl shadow-primary hover:shadow-lg transition-all duration-300 hover:scale-105 w-full"
                >
                  Take Another Interview
                </Button>
              </Card>
            </div>

            {/* Detailed Breakdown */}
            <div className="lg:col-span-2 space-y-6">
              <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 shadow-card">
                <h3 className="text-xl font-semibold mb-6">Performance Breakdown</h3>
                <div className="space-y-6">
                  {categories.map((category, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{category.name}</span>
                        <span className="text-primary font-semibold">{category.score}%</span>
                      </div>
                      <Progress value={category.score} className="h-2" />
                      <p className="text-sm text-muted-foreground">{category.description}</p>
                    </div>
                  ))}
                </div>
              </Card>

              <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 shadow-card">
                <h3 className="text-xl font-semibold mb-6">Detailed Feedback</h3>
                <div className="space-y-4">
                  {feedback.map((item, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border ${
                        item.type === "strength"
                          ? "bg-green-500/10 border-green-500/30"
                          : "bg-yellow-500/10 border-yellow-500/30"
                      }`}
                    >
                      <div className="flex items-center space-x-2 mb-2">
                        {item.type === "strength" ? (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        ) : (
                          <Lightbulb className="w-5 h-5 text-yellow-500" />
                        )}
                        <h4 className="font-semibold">{item.title}</h4>
                      </div>
                      <p className="text-sm text-muted-foreground">{item.description}</p>
                    </div>
                  ))}
                </div>
              </Card>

              <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 shadow-card">
                <h3 className="text-xl font-semibold mb-4">Next Steps</h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <Target className="w-6 h-6 text-primary mt-1" />
                    <div>
                      <h4 className="font-medium">Practice Technical Questions</h4>
                      <p className="text-sm text-muted-foreground">
                        Focus on system design and algorithm problems to improve technical depth.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <BookOpen className="w-6 h-6 text-primary mt-1" />
                    <div>
                      <h4 className="font-medium">Study Behavioral Techniques</h4>
                      <p className="text-sm text-muted-foreground">
                        Learn the STAR method for better storytelling in behavioral questions.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Presentation className="w-6 h-6 text-primary mt-1" />
                    <div>
                      <h4 className="font-medium">Mock Interviews</h4>
                      <p className="text-sm text-muted-foreground">
                        Schedule regular practice sessions to build confidence.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;