import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowRight, Code, Cpu, FileCode, HardDrive, Layers, Mic, Settings } from "lucide-react"

export default function LatentSyncUpgradePlan() {
  return (
    <div className="container mx-auto p-4 max-w-5xl">
      <h1 className="text-3xl font-bold mb-6">LatentSync Upgrade Plan</h1>

      <Tabs defaultValue="resolution">
        <TabsList className="grid w-full grid-cols-2 mb-8">
          <TabsTrigger value="resolution">Higher Resolution (512+)</TabsTrigger>
          <TabsTrigger value="whisper">Better Whisper Integration</TabsTrigger>
        </TabsList>

        <TabsContent value="resolution" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Resolution Upgrade Components
              </CardTitle>
              <CardDescription>Key components that need modification to support 512+ resolution</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <UpgradeCard
                  icon={<FileCode className="h-5 w-5" />}
                  title="Model Architecture"
                  description="Modify the UNet architecture to handle higher resolution inputs"
                  file="latentsync/models/unet.py"
                />
                <UpgradeCard
                  icon={<HardDrive className="h-5 w-5" />}
                  title="Memory Optimization"
                  description="Implement gradient checkpointing and mixed precision training"
                  file="latentsync/train.py"
                />
                <UpgradeCard
                  icon={<Cpu className="h-5 w-5" />}
                  title="Batch Processing"
                  description="Add dynamic batch sizing based on available VRAM"
                  file="latentsync/utils/batch_utils.py"
                />
                <UpgradeCard
                  icon={<Settings className="h-5 w-5" />}
                  title="Configuration Updates"
                  description="Add new configuration options for higher resolution processing"
                  file="configs/high_res_config.yaml"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="whisper" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mic className="h-5 w-5" />
                Whisper Model Upgrade
              </CardTitle>
              <CardDescription>Improvements to audio processing for better mouth articulation</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <UpgradeCard
                  icon={<Code className="h-5 w-5" />}
                  title="Whisper Integration"
                  description="Upgrade to Whisper Large-v3 for improved phoneme detection"
                  file="latentsync/audio/whisper_model.py"
                />
                <UpgradeCard
                  icon={<FileCode className="h-5 w-5" />}
                  title="Audio Processing"
                  description="Enhanced audio feature extraction for finer mouth movements"
                  file="latentsync/audio/processor.py"
                />
                <UpgradeCard
                  icon={<Settings className="h-5 w-5" />}
                  title="Phoneme Mapping"
                  description="Improved mapping between phonemes and visual features"
                  file="latentsync/models/phoneme_mapper.py"
                />
                <UpgradeCard
                  icon={<Cpu className="h-5 w-5" />}
                  title="Temporal Consistency"
                  description="Better temporal smoothing for natural mouth movements"
                  file="latentsync/models/temporal_layer.py"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function UpgradeCard({ icon, title, description, file }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          {icon}
          {title}
        </CardTitle>
        <CardDescription className="text-xs text-muted-foreground">{file}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm">{description}</p>
      </CardContent>
      <CardFooter>
        <Button variant="outline" size="sm" className="w-full">
          View Implementation <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  )
}
