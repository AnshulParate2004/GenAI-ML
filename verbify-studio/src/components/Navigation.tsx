import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Home, Video, BarChart3 } from "lucide-react";

const Navigation = () => {
  const navItems = [
    { path: "/", label: "Home", Icon: Home },
    { path: "/start-interview", label: "Start Interview", Icon: Video },
    { path: "/results", label: "Results", Icon: BarChart3 },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-nav-background/95 backdrop-blur-md border-b border-border">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">AI</span>
          </div>
          <span className="font-semibold text-lg">Interview Platform</span>
        </div>
        
        <div className="flex items-center space-x-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                cn(
                  "flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300",
                  "hover:bg-secondary/50 hover:scale-105",
                  isActive
                    ? "bg-gradient-primary text-primary-foreground shadow-primary"
                    : "text-nav-item hover:text-foreground"
                )
              }
            >
              <item.Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;