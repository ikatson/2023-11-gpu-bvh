// Input environment map
@group(0) @binding(0)
var inputTexture: texture_2d<f32>;

// Output irradiance map
@group(0) @binding(1)
var outputTexture: texture_storage_2d<rgba32float, write>;

// Helper function to convert from UV coordinates to direction vector
fn uvToDirection(uv: vec2<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * uv.x;
    let theta = PI * uv.y;

    let sinTheta = sin(theta);

    return vec3<f32>(
        -sinTheta * sin(phi),
        cos(theta),
        -sinTheta * cos(phi)
    );
}

// Helper function to convert from direction vector to UV coordinates
fn directionToUv(dir: vec3<f32>) -> vec2<f32> {
    let phi = atan2(dir.x, dir.z);
    let theta = acos(dir.y);

    let u = 1.0 - (phi + PI) / (2.0 * PI);
    let v = theta / PI;

    return vec2<f32>(u, v);
}

// Compute irradiance by sampling the environment map over a hemisphere
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(outputTexture);
    let pixelCoord = vec2<u32>(global_id.xy);

    // Get normalized texture coordinates
    let uv = (vec2<f32>(pixelCoord) + 0.5) / vec2<f32>(dimensions);

    // Get direction vector from uv coordinates
    let normal = uvToDirection(uv);

    // Create a local coordinate system around the normal
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let tangent = normalize(cross(up, normal));
    let bitangent = normalize(cross(normal, tangent));

    // Sample parameters
    let sampleDelta = 0.025;
    var numSamples = 0u;
    var irradiance = vec3<f32>(0.0);

    // Monte Carlo integration of the irradiance over the hemisphere
    for (var phi = 0.0; phi < 2.0 * PI; phi += sampleDelta) {
        for (var theta = 0.0; theta < 0.5 * PI; theta += sampleDelta) {
            // Spherical to cartesian in local space
            let sampleVec = vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            );

            // Convert to world space
            let sampleDir = sampleVec.x * tangent + sampleVec.y * bitangent + sampleVec.z * normal;

            // Convert direction to uv coordinates
            let sampleUv = directionToUv(sampleDir);

            // Sample the environment map
            let size = textureDimensions(inputTexture);
            let texelCoord = vec2<i32>(sampleUv * vec2<f32>(size));
            let color = textureLoad(inputTexture, texelCoord, 0).rgb;

            // Add to irradiance weighted by the cosine factor
            irradiance += color * cos(theta) * sin(theta);
            numSamples += 1u;
        }
    }

    // Normalize and scale the irradiance
    if (numSamples > 0u) {
        irradiance = irradiance * (PI / f32(numSamples));
    }

    // Write the irradiance value to the output texture
    textureStore(outputTexture, pixelCoord, vec4<f32>(irradiance, 1.0));
}
