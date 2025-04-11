import { Progress } from "@/components/ui/progress"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, FileCode, Layers, Mic } from "lucide-react"

export default function ImplementationProgress() {
  return (
    <div className="container mx-auto p-4 max-w-5xl">
      <h1 className="text-3xl font-bold mb-6">LatentSync Upgrade Implementation</h1>
      <p className="text-gray-500 mb-8">Implementing high-resolution support and enhanced Whisper integration</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <ProgressCard
          title="High Resolution Support"
          description="Upgrading to 512+ resolution"
          progress={40}
          icon={<Layers className="h-5 w-5" />}
        />
        <ProgressCard
          title="Whisper Integration"
          description="Enhanced audio processing for better articulation"
          progress={35}
          icon={<Mic className="h-5 w-5" />}
        />
      </div>

      <div className="space-y-6">
        <FileCard
          filename="configs/high_res_config.yaml"
          status="completed"
          description="Configuration for high-resolution processing"
        />
        <FileCard
          filename="latentsync/models/unet.py"
          status="in-progress"
          description="Modified UNet architecture for higher resolution"
        />
        <FileCard
          filename="latentsync/audio/whisper_model.py"
          status="completed"
          description="Enhanced Whisper model integration"
        />
        <FileCard
          filename="latentsync/audio/processor.py"
          status="in-progress"
          description="Improved audio processing pipeline"
        />
      </div>
    </div>
  )
}

function ProgressCard({ title, description, progress, icon }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          {icon}
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <Progress value={progress} className="h-2" />
          <p className="text-sm text-right text-gray-500">{progress}% complete</p>
        </div>
      </CardContent>
    </Card>
  )
}

function FileCard({ filename, status, description }) {
  return (
    <Card>
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg flex items-center gap-2">
            <FileCode className="h-5 w-5" />
            {filename}
          </CardTitle>
          <CardDescription>{description}</CardDescription>
        </div>
        <Badge variant={status === "completed" ? "default" : "outline"} className="ml-auto">
          {status === "completed" ? (
            <span className="flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" /> Complete
            </span>
          ) : (
            "In Progress"
          )}
        </Badge>
      </CardHeader>
    </Card>
  )
}
