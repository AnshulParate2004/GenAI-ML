import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";
import { Bot, Video, BarChart3, Play } from "lucide-react";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen pt-16 bg-gradient-hero">
      <div className="container mx-auto px-6 py-20">
        {/* Hero Section */}
        <div className="text-center max-w-4xl mx-auto mb-20">
          <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">
            AI-Powered Interview Platform
          </h1>
          <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
            Experience the future of interviews with our advanced AI system. 
            Get real-time feedback, comprehensive analysis, and personalized insights.
          </p>
          <Button
            onClick={() => navigate("/start-interview")}
            className="bg-gradient-primary hover:opacity-90 text-primary-foreground px-8 py-6 text-lg font-semibold rounded-xl shadow-primary hover:shadow-lg transition-all duration-300 hover:scale-105"
          >
            <Play className="w-5 h-5 mr-2" />
            Start Your Interview
          </Button>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/70 transition-all duration-300 hover:scale-105 shadow-card">
            <Bot className="w-12 h-12 text-primary mb-4" />
            <h3 className="text-xl font-semibold mb-3">AI-Driven Analysis</h3>
            <p className="text-muted-foreground">
              Advanced AI analyzes your responses, body language, and communication skills in real-time.
            </p>
          </Card>

          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/70 transition-all duration-300 hover:scale-105 shadow-card">
            <Video className="w-12 h-12 text-primary mb-4" />
            <h3 className="text-xl font-semibold mb-3">Video & Audio Recording</h3>
            <p className="text-muted-foreground">
              High-quality recording capabilities to capture every detail of your interview performance.
            </p>
          </Card>

          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/70 transition-all duration-300 hover:scale-105 shadow-card">
            <BarChart3 className="w-12 h-12 text-primary mb-4" />
            <h3 className="text-xl font-semibold mb-3">Detailed Insights</h3>
            <p className="text-muted-foreground">
              Comprehensive reports with actionable feedback to improve your interview skills.
            </p>
          </Card>
        </div>

        {/* Stats Section */}
        <div className="mt-20 text-center">
          <div className="grid md:grid-cols-3 gap-8 max-w-3xl mx-auto">
            <div className="space-y-2">
              <div className="text-4xl font-bold text-primary">98%</div>
              <div className="text-muted-foreground">Accuracy Rate</div>
            </div>
            <div className="space-y-2">
              <div className="text-4xl font-bold text-primary">10k+</div>
              <div className="text-muted-foreground">Interviews Conducted</div>
            </div>
            <div className="space-y-2">
              <div className="text-4xl font-bold text-primary">4.9/5</div>
              <div className="text-muted-foreground">User Satisfaction</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;