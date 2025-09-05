import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "@/hooks/use-toast";
import { Mic, Bot, Circle, Square } from "lucide-react";

interface Message {
  id: string;
  type: "bot" | "user";
  content: string;
  timestamp: Date;
}

const StartInterview = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isInterviewStarted, setIsInterviewStarted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentBotAudio, setCurrentBotAudio] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const addMessage = (type: "bot" | "user", content: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      type,
      content,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const startInterview = async () => {
    try {
      const response = await fetch("http://localhost:8000/start_interview", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to start interview");
      }

      // Backend returns audio file directly, create blob URL
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Play the bot's first audio
      setCurrentBotAudio(audioUrl);
      addMessage("bot", "Tell me about yourself.");
      
      setIsInterviewStarted(true);
      toast({
        title: "Interview Started",
        description: "Your interview has begun. Listen to the question and click record to respond.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to start interview. Please ensure your backend is running.",
        variant: "destructive",
      });
    }
  };

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: true 
      });
      
      recordedChunksRef.current = [];
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9,opus'
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const recordedBlob = new Blob(recordedChunksRef.current, {
          type: 'video/webm'
        });
        await submitAnswer(recordedBlob);
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      toast({
        title: "Recording Started",
        description: "Speak your answer clearly.",
      });
    } catch (error) {
      toast({
        title: "Recording Error",
        description: "Could not access microphone/camera.",
        variant: "destructive",
      });
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const submitAnswer = async (audioVideoBlob: Blob) => {
    try {
      const formData = new FormData();
      formData.append("audio_video", audioVideoBlob, "answer.webm");

      const response = await fetch("http://localhost:8000/submit_answer", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to submit answer");
      }

      // Check if response is JSON or binary audio
      const contentType = response.headers.get("content-type");
      
      if (contentType && contentType.includes("application/json")) {
        // Handle JSON response
        const data = await response.json();
        addMessage("user", data.transcription || "Audio/Video response submitted");
        
        if (data.audio_url) {
          setCurrentBotAudio(data.audio_url);
          addMessage("bot", data.message || "Next question...");
        }
      } else {
        // Handle binary audio response
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        addMessage("user", "Audio/Video response submitted");
        setCurrentBotAudio(audioUrl);
        addMessage("bot", "Next question...");
      }

      toast({
        title: "Answer Submitted",
        description: "Your response has been processed.",
      });
    } catch (error) {
      toast({
        title: "Submission Error",
        description: "Failed to submit your answer.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="min-h-screen pt-16 bg-background">
      <div className="container mx-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold mb-8 text-center bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">
            Interview Session
          </h1>

          {!isInterviewStarted ? (
            <Card className="p-12 text-center bg-card/50 backdrop-blur-sm border-border/50 shadow-card">
              <Mic className="w-16 h-16 text-primary mx-auto mb-6" />
              <h2 className="text-2xl font-semibold mb-4">Ready to Start?</h2>
              <p className="text-muted-foreground mb-8 max-w-md mx-auto">
                Click the button below to begin your interview. Make sure your microphone and camera are working properly.
              </p>
              <Button
                onClick={startInterview}
                className="bg-gradient-primary hover:opacity-90 text-primary-foreground px-8 py-4 text-lg font-semibold rounded-xl shadow-primary hover:shadow-lg transition-all duration-300 hover:scale-105"
              >
                Start Interview
              </Button>
            </Card>
          ) : (
            <div className="space-y-8">
              {/* Audio Player for Bot Responses */}
              {currentBotAudio && (
                <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
                  <div className="flex items-center space-x-4">
                    <Bot className="w-8 h-8 text-primary" />
                    <div className="flex-1">
                      <h3 className="font-semibold mb-2">AI Interviewer</h3>
                      <audio
                        ref={audioRef}
                        src={currentBotAudio}
                        controls
                        className="w-full"
                        autoPlay
                      />
                    </div>
                  </div>
                </Card>
              )}

              {/* Recording Controls */}
              <Card className="p-8 text-center bg-card/50 backdrop-blur-sm border-border/50">
                <div className="space-y-6">
                  <div className="w-16 h-16 mx-auto flex items-center justify-center">
                    {isRecording ? (
                      <Circle className="w-16 h-16 text-red-500 fill-current animate-pulse" />
                    ) : (
                      <Mic className="w-16 h-16 text-primary" />
                    )}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-2">
                      {isRecording ? "Recording Your Response..." : "Ready to Record"}
                    </h3>
                    <p className="text-muted-foreground">
                      {isRecording 
                        ? "Speak clearly and click stop when finished"
                        : "Click the button to start recording your answer"
                      }
                    </p>
                  </div>
                  <Button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-300 hover:scale-105 ${
                      isRecording
                        ? "bg-destructive hover:bg-destructive/90 text-destructive-foreground"
                        : "bg-gradient-primary hover:opacity-90 text-primary-foreground shadow-primary hover:shadow-lg"
                    }`}
                  >
                    {isRecording ? (
                      <>
                        <Square className="w-5 h-5 mr-2" />
                        Stop Recording
                      </>
                    ) : (
                      <>
                        <Mic className="w-5 h-5 mr-2" />
                        Start Recording
                      </>
                    )}
                  </Button>
                </div>
              </Card>

              {/* Conversation History */}
              {messages.length > 0 && (
                <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
                  <h3 className="text-xl font-semibold mb-4">Conversation History</h3>
                  <div className="space-y-4 max-h-80 overflow-y-auto">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex space-x-3 ${
                          message.type === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        <div
                          className={`max-w-xs px-4 py-2 rounded-lg ${
                            message.type === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-secondary text-secondary-foreground"
                          }`}
                        >
                          <div className="text-sm">
                            {message.type === "user" ? "You" : "AI Interviewer"}
                          </div>
                          <div>{message.content}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StartInterview;